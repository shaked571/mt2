import argparse
import os
import random
from typing import List
from timeit import default_timer as timer

import numpy as np
import torch.nn as nn

from data_sets import TranslationDataSet
from models import EncoderVanilla, DecoderVanilla, DecoderAttention, Seq2Seq
from vocab import Vocab
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import sacrebleu

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker

all_eval_times = []


def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() == 'cuda':
        torch.cuda.manual_seed_all(seed)


def pad_collate(batch):
    (ss, tt) = zip(*batch)
    source_lens = [len(sent) for sent in ss]
    target_lens = [len(sent) for sent in tt]

    xx_pad = pad_sequence(ss, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(tt, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, source_lens, target_lens


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


class Trainer:
    def __init__(self, model: nn.Module, train_data: Dataset, dev_data: Dataset, source_vocab: Vocab,
                 target_vocab: Vocab, train_batch_size=2, lr=0.002, out_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.part = 1 if type(model.decoder) == DecoderVanilla else 2
        self.model = model
        self.dev_batch_size = 1
        self.n_epochs = 10
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.train_data = DataLoader(train_data, batch_size=train_batch_size, collate_fn=pad_collate)
        self.dev_data = DataLoader(dev_data, batch_size=self.dev_batch_size, collate_fn=pad_collate)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.target_vocab.PAD_IDX)
        self.model.to(self.device)
        self.model_args = {"part": self.part, "lr": lr, "batch_size": train_batch_size,
                           "embed_size": self.model.encoder.embed_size,
                           "hidden_dim": self.model.encoder.hidden_size * 2, "dropout": self.model.decoder.dropout.p,
                           "n_layers": self.model.decoder.n_layers}
        if out_path is None:
            out_path = self.suffix_run()

        self.saved_model_path = f"{out_path}.bin"
        self.writer = SummaryWriter(log_dir=f"tensor_board/{out_path}")
        self.best_model = None
        self.best_score = 0

    def train(self, dump_vis=False):
        for epoch in range(self.n_epochs):
            print(f"start epoch: {epoch + 1}")
            train_loss = 0.0
            step_loss = 0
            self.model.train()  # prep model for training

            for step, (source, target, source_lens, target_lens) in tqdm(enumerate(self.train_data),
                                                                         total=len(self.train_data)):
                source = source.to(self.device)
                target = target.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                self.model.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(source, target, train=True)
                output_dim = output.shape[-1]  # Handling batch size N > 1
                output = output.view(-1, output_dim)  # Handling batch size N > 1

                # calculate the loss
                loss = self.loss_func(output, target.view(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # print(self.model.decoder.out_linear.weight)
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item() * target.size(0)
                step_loss += loss.item() * target.size(0)
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch + 1)
            if dump_vis:
                self.dump_vis_attention(epoch + 1)
            else:
                self.evaluate_model(epoch + 1, "epoch", self.dev_data)
        if len(all_eval_times) > 0:
            print(f"average time to evaluate: {sum(all_eval_times) / len(all_eval_times)}")

    def evaluate_model(self, step, stage, data_set, load_model=False, write=True):
        if load_model:
            checkpoint = torch.load(self.saved_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
            self.model.load_state_dict(checkpoint)
        with torch.no_grad():
            bleu_score, loss, start_eval_time = self.eval_loop(data_set, step)
            end_eval_time = timer()
            time_to_evaluate = end_eval_time - start_eval_time
            all_eval_times.append(time_to_evaluate)
            print(f"Eval time for stage {stage}-step {step} is: {time_to_evaluate} ")

            if write:
                print(f'bleu_score/dev_{stage}: {bleu_score}')
                self.writer.add_scalar(f'bleu_score/dev_{stage}', bleu_score, step)
                self.writer.add_scalar(f'Loss/dev_{stage}', loss, step)
                if bleu_score > self.best_score:
                    self.best_score = bleu_score
                    print("best score: ", self.best_score)
                    torch.save(self.model.state_dict(), self.saved_model_path)
            else:
                print(f'bleu_score/{stage}: {bleu_score}')
        self.model.train()

    def eval_loop(self, data_set, step):
        self.model.eval()
        loss = 0
        prediction = []
        all_target = []
        start_eval_time = timer()
        for eval_step, (source, target, source_lens, target_lens) in tqdm(enumerate(data_set), total=len(data_set),
                                                                          desc=f"dev step {step} loop"):
            source = source.to(self.device)
            target = target.to(self.device)
            output = self.model(source, target, train=False)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            target = target.view(-1)
            padding_len = abs(len(target) - len(output))
            if len(target) > len(output):
                out_pad = torch.zeros((padding_len, output_dim), device=self.device)
                output = torch.cat((output, out_pad))
            elif len(output) > len(target):
                target_pad = torch.zeros(padding_len, device=self.device)
                target = torch.cat((target, target_pad)).type(torch.int64)

            loss = self.loss_func(output, target)
            loss += loss.item()
            predicted = torch.argmax(output, dim=1)
            prediction.append(predicted.view(-1).tolist())
            all_target.append(target.view(-1).tolist())
        bleu_score = self.bleu_score(prediction, all_target)
        return bleu_score, loss, start_eval_time

    def suffix_run(self):
        res = ""
        for k, v in self.model_args.items():
            res += f"{k}_{v}_"
        res = res.strip("_")
        return res

    def test(self, test_df):
        test = DataLoader(test_df, batch_size=self.dev_batch_size, collate_fn=pad_collate)
        self.model.load_state_dict(torch.load(self.saved_model_path))
        self.model.eval()
        prediction = []
        for eval_step, (source, target_dummy, source_lens, target_lens) in tqdm(enumerate(test), total=len(test),
                                                                                desc=f"test data"):
            source = source.to(self.device)
            output = self.model(source, target_dummy, train=False)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            predicted = torch.argmax(output, dim=1)
            prediction.append(predicted.view(-1).tolist())
        return [self.target_vocab.i2token[i] for i in prediction]

    def bleu_score(self, predict: List, target: List):
        predict, target = self.get_un_padded_samples(predict, target)

        bleu = sacrebleu.corpus_bleu(predict, target)
        return bleu.score

    def get_un_padded_samples(self, predict, target):
        no_pad_predict = []
        no_pad_target = []

        for p_sen, t_sen in zip(predict, target):
            unpad_t = []
            unpad_p = []
            finish_pred = False
            finish_target = False
            for p, t in zip(p_sen[1:], t_sen[1:]):
                if not finish_target:
                    if t == self.target_vocab.END_IDX:
                        finish_target = True
                        continue
                    unpad_t.append(self.target_vocab.i2token[t])
                if not finish_pred:
                    if p == self.target_vocab.END_IDX:
                        finish_pred = True
                        continue
                    unpad_p.append(self.target_vocab.i2token[p])

            no_pad_predict.append(" ".join(unpad_p))
            no_pad_target.append(" ".join(unpad_t))
        return no_pad_predict, [no_pad_target]

    def dump_vis_attention(self, epoch):
        # Take the integer value of <sos> from the target vocabulary.
        source, target, source_len, target_len = list(self.dev_data)[71]
        with torch.no_grad():
            self.model(source.to(self.device), target.to(self.device), train=False, target_vocab=self.target_vocab, epoch=epoch)






def parse_arguments():
    p = argparse.ArgumentParser(description='Hyper parameters')
    p.add_argument('decoder', help="decoder type: {v, a} when 'v' is DecoderVanilla and 'a' is AttentionDecoder.")
    p.add_argument('-b', '--batch_size', type=int, default=4, help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.002, help='initial learning rate')
    p.add_argument('-hs', '--hidden_size', type=int, default=256, help='number of epochs for train')
    p.add_argument('-e', '--embed_size', type=int, default=128, help='embedding size')
    p.add_argument('-p', '--dropout', type=float, default=0.3, help='dropout precentage')
    p.add_argument('-n', '--n_layers', type=int, default=2, help='number of layers in rnn')
    p.add_argument('-o', '--out_path', type=str, help='number of epochs for train')

    return p.parse_args()


def main():
    set_seed(42)

    args = parse_arguments()
    hidden_size = args.hidden_size
    embed_size = args.embed_size

    data_path = "data/{}.{}"

    source_vocab = Vocab(data_path.format("train", "src"))
    target_vocab = Vocab(data_path.format("train", "trg"))

    encoder = EncoderVanilla(vocab_size=source_vocab.vocab_size, embed_size=embed_size, hidden_size=hidden_size,
                             n_layers=args.n_layers, dropout=args.dropout)
    if args.decoder == 'v':
        decoder = DecoderVanilla(vocab_size=target_vocab.vocab_size, embed_size=embed_size, hidden_size=hidden_size,
                                 n_layers=args.n_layers, dropout=args.dropout)
    elif args.decoder == 'a':
        decoder = DecoderAttention(vocab_size=target_vocab.vocab_size, embed_size=embed_size, hidden_size=hidden_size,
                                   n_layers=args.n_layers, dropout=args.dropout)
    else:
        raise ValueError("Support decoder type: {v, a} when 'v' is DecoderVanilla and 'a' is AttentionDecoder.")

    model = Seq2Seq(encoder, decoder)
    print(model)
    train_df = TranslationDataSet(source=data_path.format("train", "src"), target=data_path.format("train", "trg"),
                                  source_vocab=source_vocab, target_vocab=target_vocab)
    dev_df = TranslationDataSet(source=data_path.format("dev", "src"), target=data_path.format("dev", "trg"),
                                source_vocab=source_vocab, target_vocab=target_vocab)

    trainer = Trainer(model=model, train_data=train_df, dev_data=dev_df, source_vocab=source_vocab,
                      target_vocab=target_vocab, lr=args.lr, train_batch_size=args.batch_size, out_path=args.out_path)
    trainer.train()


if __name__ == '__main__':
    main()
