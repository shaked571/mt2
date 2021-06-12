from torch.utils.data import DataLoader

from data_sets import TranslationDataSet
from trainer import Trainer, set_seed, pad_collate
from part1_train import model, data_path, source_vocab, target_vocab, batch_size, lr


def main():
    set_seed(42)
    print(model)
    train_df = TranslationDataSet(source=data_path.format("train", "src"), target=data_path.format("train", "trg"),
                                  source_vocab=source_vocab, target_vocab=target_vocab)
    dev_df = TranslationDataSet(source=data_path.format("dev", "src"), target=data_path.format("dev", "trg"),
                                source_vocab=source_vocab, target_vocab=target_vocab)

    trainer = Trainer(model=model, train_data=train_df, dev_data=dev_df, source_vocab=source_vocab,
                      target_vocab=target_vocab, lr=lr, train_batch_size=batch_size, out_path="part1_model")


    test_df = TranslationDataSet(source=data_path.format("test", "src"), target=data_path.format("test", "trg"),
                                source_vocab=source_vocab, target_vocab=target_vocab)

    test_dateset = DataLoader(test_df, batch_size=trainer.dev_batch_size, collate_fn=pad_collate)

    trainer.evaluate_model(1, "test", test_dateset, load_model=True, write=False)


if __name__ == '__main__':
    main()
