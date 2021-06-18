from data_sets import TranslationDataSet
from models import EncoderVanilla, DecoderAttention, Seq2Seq
from trainer import Trainer, set_seed
from vocab import Vocab
hidden_size = 128
embed_size = 256
n_layers = 2
dropout = 0.3
lr = 0.002
batch_size = 8
data_path = "data/{}.{}"

source_vocab = Vocab(data_path.format("train", "src"))
target_vocab = Vocab(data_path.format("train", "trg"))
encoder = EncoderVanilla(vocab_size=source_vocab.vocab_size, embed_size=embed_size, hidden_size=hidden_size,
                         n_layers=n_layers, dropout=dropout)
decoder = DecoderAttention(vocab_size=target_vocab.vocab_size, embed_size=embed_size, hidden_size=hidden_size,
                               n_layers=n_layers, dropout=dropout)
model = Seq2Seq(encoder, decoder)
print(model)


def main():
    set_seed(42)
    train_df = TranslationDataSet(source=data_path.format("train", "src"), target=data_path.format("train", "trg"),
                                  source_vocab=source_vocab, target_vocab=target_vocab)
    dev_df = TranslationDataSet(source=data_path.format("dev", "src"), target=data_path.format("dev", "trg"),
                                source_vocab=source_vocab, target_vocab=target_vocab)

    trainer = Trainer(model=model, train_data=train_df, dev_data=dev_df, source_vocab=source_vocab,
                      target_vocab=target_vocab, lr=lr,train_batch_size=batch_size, out_path="part2_model")
    trainer.train()


if __name__ == '__main__':
    main()
