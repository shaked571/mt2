from data_sets import TranslationDataSet
from trainer import Trainer, set_seed
from part2_train import model, data_path, source_vocab, target_vocab, batch_size, lr


def main():
    set_seed(42)
    print(model)
    train_df = TranslationDataSet(source=data_path.format("train", "src"), target=data_path.format("train", "trg"),
                                  source_vocab=source_vocab, target_vocab=target_vocab)
    dev_df = TranslationDataSet(source=data_path.format("dev", "src"), target=data_path.format("dev", "trg"),
                                source_vocab=source_vocab, target_vocab=target_vocab)

    trainer = Trainer(model=model, train_data=train_df, dev_data=dev_df, source_vocab=source_vocab,
                      target_vocab=target_vocab, lr=lr, train_batch_size=batch_size, out_path="part2_model")


    trainer.train(dump_vis=True)


if __name__ == '__main__':
    main()
