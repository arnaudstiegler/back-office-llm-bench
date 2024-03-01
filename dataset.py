from dataclasses import dataclass
from typing import Dict, Any

from datasets import load_dataset


@dataclass
class Sample:
    id: int
    task_definition: str
    task_input: str
    answer: str
    prompt: str


class FinanceTasksDataset:
    def __init__(self) -> None:
        self.dataset = load_dataset("AdaptLLM/finance-tasks", "ConvFinQA")["test"]
        self.task_definition = """
            Answer the question and return the answer in a json that contains one key called "answer"
            and the value must be the unique string that is the answer to that question.
            If there are multiple questions in the input, only answer the last question
        """

    def __getitem__(self, item: int) -> Sample:
        sample = self.dataset[item]
        return Sample(
            id=item,
            task_input=sample["input"],
            task_definition=self.task_definition,
            answer=sample["label"],
            prompt=self.task_definition + "  \n  " + sample["input"],
        )


class OpenMathDataset:
    def __init__(self):
        self.dataset = load_dataset("nvidia/OpenMathInstruct-1")["validation"]
        self.task_definition = self.get_task_definition

    @property
    def get_task_definition(self) -> str:
        return """First reason and find the solution to the question. Then format the answer of the question 
        as a json dictionnary with a key "answer" and the value the corresponding numerical answer 
        to the question. Make sure there is one and only one json dict in your answer and that nothing
        else is formatted similarly (i will parse that answer for a json)"""

    def __getitem__(self, item: int) -> Sample:
        sample = self.dataset[item]
        return Sample(
            id=item,
            task_input=sample["question"],
            task_definition=self.task_definition,
            answer=sample["expected_answer"],
            prompt=sample["question"] + "  \n  " + self.task_definition,
        )

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    data = OpenMathDataset()
    print(data[0])
