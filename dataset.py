from datasets import load_dataset


class FinanceTasksDataset:
    def __init__(self):
        self.dataset = load_dataset("AdaptLLM/finance-tasks")

    def __getitem__(self, item):
        return self.dataset['train'][item]


if __name__ == '__main__':
    data = FinanceTasksDataset()
    print(data[0])