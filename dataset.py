import csv
from dataclasses import dataclass
from typing import Dict, List, Generator, Any

from datasets import load_dataset
from datasets import Dataset as HF_Dataset
from torch.utils.data import Dataset
import json

@dataclass
class Sample:
    id: int
    task_definition: str
    task_input: str
    answer: str  # TODO: maybe reformat so that it's a json?
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
            task_definition=self.get_task_definition,
            answer=sample["expected_answer"],
            prompt=sample["question"] + "  \n  " + self.get_task_definition,
        )

    def __len__(self):
        # Limit it to 2k samples on purpose
        return 2000


class KleisterNdaDataset(Dataset):
    def __init__(self, json_mode: bool):
        self.documents = self.parse_documents_tsv("./data/kleister_nda/in.tsv")
        self.ground_truth = self.parse_ground_truth_tsv(
            "./data/kleister_nda/expected.tsv"
        )
        self.json_mode = json_mode

    @property
    def get_task_definition(self) -> str:
        # If json mode is on, the model cannot do the COT and instructions need to reflect that
        return """
        The text above is the transcription of an NDA. 

        # Extraction:
        There are up to 6 attributes to be extracted from the transcription:

        effective_date - date in YYYY-MM-DD format, at which point the contract is legally binding,
        jurisdiction - under which state or country jurisdiction is the contract signed,
        party - party or parties of the contract,
        term - length of the legal contract as expressed in the document.
        Note that party usually occur more than once.

        # Normalization:
        The expected pieces of information were normalized to some degree:

        - in attribute values, all spaces and colons : were replaced with an underscores _,
        - all expected dates should be returned in YYYY-MM-DD format, values for attribute term are 
        normalized with the same original units e.g. eleven months is changed to 11_months; all of 
        them are in the same format: {number}_{units}.
        - for jurisdiction, only return the state name and nothing else. For instance, return "California" instead of "State_of_California"
        - if a field is not present in the document, do not add the key/value pair for it in the output
        
        # Output Format:
        The output has to be a valid json with the following format:
        {"effective_date": "value", "jurisdiction":"value", "party":["value_1", "value_2", ...], "term":"value"}

        So for instance:
        {"effective_date": "2020-01-12", "jurisdiction": "Utah", "party": ["Bill_Gates", "Coca_Cola_Inc."]}

        You can reason and explain your choices before returning the JSON object, you just have to have that json object in the output.
        """

    @staticmethod
    def read_tsv(path: str) -> Generator[List, Any, None]:
        with open(path, "r") as tsvfile:
            reader = csv.reader(tsvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
            for item in reader:
                yield item

    def parse_documents_tsv(self, path: str) -> List[Dict[str, str]]:
        documents = []
        for item in self.read_tsv(path):
            doc_name = item[0]
            transcription = item[5]
            documents.append({"filename": doc_name, "transcription": transcription})
        return documents

    @staticmethod
    def string_to_json_dict(input_string: str) -> dict:
        # Split the input string into key-value pairs
        pairs = input_string.split(" ")
        json_dict = {}
        for pair in pairs:
            key, value = pair.split("=")
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
            ground_truth.append(self.string_to_json_dict(item[0]))
        return ground_truth

    def __getitem__(self, item: int):
        document = self.documents[item]
        ground_truth = self.ground_truth[item]
        sample = Sample(
            id=item,
            task_definition="",
            task_input=document["transcription"],
            answer=json.dumps(ground_truth),
            prompt="Transcription: \n"
            + document["transcription"]
            + " \n "
            + self.get_task_definition,
        )
        return sample

    def __len__(self) -> int:
        return len(self.ground_truth)


class MultiHopQADataset(Dataset):
    def __init__(self, json_mode: bool):
        self.dataset = self.filter_samples(
            load_dataset("khaimaitien/qa-expert-multi-hop-qa-V1.0")["train"]
        )
        self.json_mode = json_mode

    @staticmethod
    def filter_samples(dataset: HF_Dataset) -> List[Dict[str, Any]]:
        # only keep the questions with a short form answer
        return [sample for sample in dataset if sample["answer"]]

    @property
    def get_task_definition(self) -> str:
        prompt = """
        Answer the question and return a JSON object with a key "answer" and the value being the answer
        to the question. Do not put any explanation of any sort in the Json object, only the actual
        answer to the question.
        
        For instance, if the question is: what is the capital of the country with the largest GDP in the world?
        The output would be:
        {"answer": "USA"}
        Typically, the answer should only be a few words and not a sentence.
        
        """
        if not self.json_mode:
            prompt += " Think step by step. First write the reasoning to get to the answer, then write the Json object containing only the answer"
        return prompt

    def __getitem__(self, item: int) -> Sample:
        sample = self.dataset[item]
        return Sample(
            id=item,
            task_input=sample["question"],
            task_definition=self.get_task_definition,
            answer=sample["answer"],
            prompt=sample["question"] + "  \n  " + self.get_task_definition,
        )

    def __len__(self):
        # Limit it to 2k samples on purpose
        return 500


if __name__ == "__main__":
    data = MultiHopQADataset(json_mode=False)
    print(data[0])
