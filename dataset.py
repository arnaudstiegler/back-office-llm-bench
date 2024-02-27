from dataclasses import dataclass
from typing import Dict, Any

from datasets import load_dataset


@dataclass
class Sample:
    input: str
    answer: str


class FinanceTasksDataset:
    def __init__(self):
        self.dataset = load_dataset("AdaptLLM/finance-tasks", 'ConvFinQA')['test']

    def __getitem__(self, item):
        sample = self.dataset[item]
        return Sample(
            input=sample['input'],
            answer=sample['label']
        )


if __name__ == '__main__':
    data = FinanceTasksDataset()
    print(data[0])
