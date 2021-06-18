from torch.utils.data import Dataset
from typing import List, Optional
from dataclasses import dataclass

import torch
from vocab import Vocab


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid:  Unique id for the example.
        words: list. The words of the sequence.
        label: (Optional) str. The label of the middle word in the window
    """
    guid: int
    source: List[str]
    target: Optional[List[str]]


class TranslationDataSet(Dataset):
    def __init__(self,
                 source: str,
                 target: str,
                 source_vocab: Vocab,
                 target_vocab: Vocab,
                 ):
        self.source_path = source
        self.target_path = target
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.data = self.get_examples()

    def get_examples(self):
        examples = []
        with open(self.source_path, mode="r") as f:
            source = f.readlines()
        with open(self.target_path, mode="r") as f:
            target = f.readlines()

        guid_index = 0
        for source_line, target_line in zip(source, target):
            if source == "" or source == "\n":
                continue
            guid_index += 1
            curr_source = [self.source_vocab.START] + source_line.strip().split() + [self.source_vocab.END]
            curr_target = [self.target_vocab.START] + target_line.strip().split() + [self.target_vocab.END]
            example = InputExample(guid=guid_index, source=curr_source, target=curr_target)
            examples.append(example)
        return examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source = self.data[index].source
        target = self.data[index].target

        source_tensor = torch.tensor([self.source_vocab.get_word_index(word) for word in source]).to(torch.int64)
        target_tensor = torch.tensor([self.target_vocab.get_word_index(word) for word in target]).to(torch.int64)

        return source_tensor, target_tensor

