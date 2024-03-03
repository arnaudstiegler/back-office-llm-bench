import csv
from dataclasses import dataclass
from typing import Dict, List, Generator, Any

from datasets import load_dataset
from torch.utils.data import Dataset


@dataclass
class Sample:
    id: int
    task_definition: str
    task_input: str
    answer: str  #TODO: maybe reformat so that it's a json?
    prompt: str


class FinanceTasksDataset(Dataset):
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


class OpenMathDataset(Dataset):
    def __init__(self, json_mode: bool):
        self.dataset = load_dataset("nvidia/OpenMathInstruct-1")["validation"]
        self.json_mode = json_mode
        self.task_definition = self.get_task_definition

    @property
    def get_task_definition(self) -> str:
        # If json mode is on, the model cannot do the COT and instructions need to reflect that
        if self.json_mode:
            return """Return the answer to the math problem and format it as a json dictionnary with a key 
            "answer" and the value the corresponding numerical answer to the question as a string."""
        return """First reason and find the solution to the question. Then format the answer of the question 
        as a json dictionnary with a key "answer" and the value the corresponding numerical answer 
        to the question as a string. 
        For instance:
        What is 1+1?
        1+1=2, therefore the json return is: {'answer': '2'}
        
        Make sure there is one and only one json dict in your answer, that
         all keys and values are valid strings, and that nothing
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
        # Limit it to 2k samples on purpose
        return 2000


class KleisterNdaDataset(Dataset):
    def __init__(self):
        self.documents = self.parse_documents_tsv('./data/kleister_nda/in.tsv')
        self.ground_truth = self.parse_ground_truth_tsv('./data/kleister_nda/expected.tsv')

    @staticmethod
    def read_tsv(path: str) -> Generator[List, Any, None]:
        with open(path, 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for item in reader:
                yield item

    def parse_documents_tsv(self, path: str) -> List[Dict[str, str]]:
        documents = []
        for item in self.read_tsv(path):
            doc_name = item[0]
            transcription = item[5]
            documents.append(
                {'filename': doc_name,
                 'transcription': transcription
                 }
            )
        return documents

    @staticmethod
    def string_to_json_dict(input_string: str) -> dict:
        # Split the input string into key-value pairs
        pairs = input_string.split(' ')
        json_dict = {}
        for pair in pairs:
            key, value = pair.split('=')
            # Check if the key already exists and is expected to have multiple values
            if key in json_dict:
                # If it's already a list, append the new value
                if isinstance(json_dict[key], list):
                    json_dict[key].append(value)
                else:
                    # Convert it into a list with the current and new value
                    json_dict[key] = [json_dict[key], value]
            else:
                # Add the key-value pair to the dictionary
                json_dict[key] = value
        return json_dict

    def parse_ground_truth_tsv(self, path) -> List[Dict[str, str]]:
        ground_truth = []
        for item in self.read_tsv(path):
            ground_truth.append(
                self.string_to_json_dict(item[0])
            )
        return ground_truth

    def __getitem__(self, item: int):
        document = self.documents[item]
        ground_truth = self.ground_truth[item]
        sample = Sample(
            id=item,
            task_definition='',
            task_input=document['transcription'],
            answer=str(ground_truth),
            prompt=''
        )
        return sample

    def __len__(self) -> int:
        return len(self.ground_truth)


if __name__ == "__main__":
    data = KleisterNdaDataset()
    print(data[0])
