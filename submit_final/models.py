import torch.nn as nn
import torch
import numpy as np
from vocab import Vocab

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim * 2  # We are working with bi-directional
        self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, rnn_outputs, final_hidden_state):
        # rnn_output.shape:         (B, seq_len,  H)
        # final_hidden_state.shape: (B, H)
        # hidden_size may also reflect bidirectional hidden states (hidden_size = num_directions * hidden_dim)
        rnn_outputs = rnn_outputs.transpose(0, 1)
        batch_size, seq_len, _ = rnn_outputs.shape
        attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
        attn_weights = self.tanh(attn_weights)
        attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))
        attn_weights = self.softmax(attn_weights.squeeze(2))

        return attn_weights


class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, vocab_size,embed_size,
                 n_layers=1, dropout=0.2):
        super(DecoderAttention, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.num_directions = 2
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, n_layers,bidirectional=True,
                            dropout=dropout)
        self.out_linear = nn.Linear(hidden_size * 4, vocab_size)

    def forward(self, input_tok, hidden, cell, encoder_outputs):
        batch_size = input_tok.size()[0]
        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input_tok)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs

        #Get the hidden size from the layers and directions
        final_state = hidden.view(self.n_layers, self.num_directions, batch_size, self.hidden_size)[-1]
        #Because num of directions is always 2 now we need to ge both directions
        h_1, h_2 = final_state[0], final_state[1]
        # final_hidden_state = h_1 + h_2  -> Add both states (requires changes to the input size of first linear layer + attention layer)
        final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        attn_weights = self.attention(encoder_outputs, final_hidden_state)
        context = torch.bmm(encoder_outputs.transpose(0, 1).transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2) #B * (H * Num_of_dirc)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 1) # rnn_input B * (embedded + context)
        output,  (hidden, cell) = self.lstm(rnn_input.unsqueeze(0),  (hidden, cell)) #hidden ->
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0) if len(context.shape) == 3 else context #dealing with batch size 1
        output = self.out_linear(torch.cat([output, context], 1)) #
        return output, hidden, attn_weights


class EncoderVanilla(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.2):
        super(EncoderVanilla, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers,bidirectional=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source):
        embedded = self.embedding(source)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(0, 1) # make it (seq_len, batch_size, features)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell



class DecoderVanilla(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_size,
                 n_layers=1, dropout=0.5):
        super(DecoderVanilla, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.vocab_size, embed_size * 2, padding_idx=0)
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, n_layers,bidirectional=True,
                          dropout=dropout)

        self.out_linear = nn.Linear(hidden_size*2, self.vocab_size)
        self.relu = nn.ReLU()

    def forward(self, input_tok, hidden, cell):
        # input is the previous predicted word
        output = self.embedding(input_tok)
        output = self.dropout(output)
        output = self.relu(output)
        output, (hidden, cell) = self.lstm(output.unsqueeze(0), (hidden, cell))
        output = self.out_linear(output) #  N * 1 * lenght_of_vocab
        output = output.squeeze(0)
        return output, (hidden, cell)


class Seq2Seq(nn.Module):
    max_len = 50
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"


    def forward(self, src, trg, train, target_vocab=None, epoch=None):
        max_len = trg.size(1) if train or target_vocab is not None else self.max_len
        attentions = []
        batch_size = src.size(0)
        vocab_size = self.decoder.vocab_size
        #init the strat output with 1 in the start index and 0 on the rest
        start_output = np.zeros((batch_size, vocab_size))
        start_output[:, 1] = 1
        outputs = [torch.tensor(start_output,  device=self.device, dtype=torch.float64, requires_grad=True)]
        encoder_output, hidden, cell = self.encoder(src)
        output = trg[:, 0] if trg is not None else torch.ones(batch_size, device=self.device).type(torch.int) #init start token as an array - Encoder forword will transpose them
        for t in range(1, max_len):
            if type(self.decoder) == DecoderVanilla:
                output, (hidden, cell) = self.decoder(
                    output, hidden, cell)
            else:  # Attention
                output, hidden, attn_weights = self.decoder(
                    output, hidden, cell, encoder_output)
                if target_vocab is not None:
                    attentions.append(attn_weights)

            outputs.append(output)
            if train:
                output = trg[:, t]
            elif target_vocab is not None: # for visualization
                output = trg[0][t].unsqueeze(0)
            else:
                output[:, 0] = -100  # Dont allow prediction of Padding index.
                output = output.argmax(1)
                if all([i == Vocab.END_IDX for i in output.cpu().numpy()]):  # Support Dev only for batch in size of 1
                    break
        if target_vocab is not None:

            attentions = torch.stack(attentions, dim=1)
            display_attention(src, outputs, attentions, target_vocab, epoch)
        outputs = torch.stack(outputs, dim=1)
        return outputs

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker


def display_attention(sentence, translation, attention, target_vocab:Vocab, epoch):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)


    # sentence_i = ['<s>', '104', '111', '108', '100', '105', '110', '103', '46', '</s>']
    sentence_i = ['<s>', '97', '103', '97', '105', '110', '115', '116'] #97 103 97 105 110 115 116
    # sentence = ['', 'h', 'o', 'l', 'd', 'i', 'n', 'g', '.', '']
    sentence = ['', 'a', 'g', 'a', 'i', 'n', 's', 't']

    translation_res = [target_vocab.i2token[i.item()] for i in torch.stack(translation, dim=1).squeeze().argmax(1)]
    attention_f = attention.squeeze(0).cpu().detach().numpy()[:, 1:]#, 1:-1]

    cax = ax.matshow(attention_f, cmap='bone')

    ax.tick_params(labelsize=15, direction='out')
    print("res: " , translation_res)
    print("x:", [''] + [f'{t}-{i}' for t, i in zip(sentence, sentence_i)])
    ax.set_xticklabels([''] + [f'{t}-{i}' for t, i in zip(sentence, sentence_i)],
                       rotation=45)
    ax.set_yticklabels([''] + translation_res + [''])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title(f"Attention epoch {epoch}",fontdict={'fontsize': 20, 'fontweight': 'medium'})
    ax.set_xlabel(f"Source")
    ax.set_ylabel(f"Prediction")
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    plt.savefig(f"Attention_epoch_{epoch}.png")
    plt.show()
    plt.close()
