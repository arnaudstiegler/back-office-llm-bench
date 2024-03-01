import json
import os

import click

from predictors import (
    MistralOpenOrcaPredictor,
    MistralInstructPredictor,
    MAX_NEW_TOKENS,
)
from dataset import OpenMathDataset
from utils import find_and_parse_json

dir_path = os.path.dirname(os.path.realpath(__file__))
samples = json.load(open(os.path.join(dir_path, "tasks_manual.json")))["samples"]

MODEL_CHOICES = {
    "mistral-orca": MistralOpenOrcaPredictor,
    "mistral-instruct": MistralInstructPredictor,
}


@click.command()
@click.option(
    "--model",
    type=click.Choice(list(MODEL_CHOICES.keys())),
    default="mistral-instruct",
)
@click.option("--json_mode", is_flag=True)
def run_eval(model: str, json_mode: bool) -> None:
    # model_predictor_cls = MODEL_CHOICES[model]
    model_predictor = MistralInstructPredictor()
    dataset = OpenMathDataset()

    answer_correct = []
    for i in range(1, 100):
        sample = dataset[i]
        # prompt = model_predictor.format_sample_into_prompt(sample)

        from jsonformer import Jsonformer

        json_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "number"},
            },
        }
        # print(sample)
        # Jsonformer should take care of the input
        prompt = (
            "[INST] " + sample.task_input + " " + sample.task_definition + " [/INST]"
        )
        jsonformer = Jsonformer(
            model_predictor.model,
            model_predictor.tokenizer,
            json_schema,
            prompt,
            max_string_token_length=MAX_NEW_TOKENS,
        )
        generated_data = jsonformer()
        #
        print("with json_mode", generated_data)

        # TODO: should I add '[INST]' + prompt + '[/INST]'?
        # -> The answer to that is YES!
        answer = model_predictor.predict(sample)

        print("recovered", find_and_parse_json(answer))
        print("answer", sample.answer)

        if (
            find_and_parse_json(answer)
            and find_and_parse_json(answer).get("answer")
            and str(answer) == sample.answer
        ):
            answer_correct.append(1)
        else:
            answer_correct.append(0)

        if len(answer_correct) > 0:
            print(sum(answer_correct) / len(answer_correct))
        print("\n")

    # valid_json = []
    # correct = []
    # for entry in samples:
    #     sample = Sample(**entry)
    #     generated_answer = model_predictor.generate_answer(sample=sample)
    #     print(generated_answer)
    #     print('\n')
    #
    #     try:
    #         json_answer = json.loads(generated_answer)
    #         valid_json.append(1)
    #         if json_answer == sample.expected_output:
    #             correct.append(1)
    #         else:
    #             correct.append(0)
    #     except:
    #         out = find_and_parse_json(generated_answer)
    #         if out:
    #             # TODO: invalid format a priori, but could be recovered?
    #             if out == sample.expected_output:
    #                 correct.append(1)
    #             else:
    #                 correct.append(0)
    #             valid_json.append(0)
    #         else:
    #             correct.append(0)
    #             valid_json.append(0)
    #             print(f'Could not parse json from: {generated_answer}')
    #
    #     """
    #     Use jsonformer to force json output
    #     For eval:
    #     if json_expected: try json.loads first. If not working, try retrieve a json from the output and load it
    #     Verify the keys
    #     if not json_expected: try to match only the output, else try to find the answer in the output
    #     """
    # print(valid_json)
    # print(correct)
    # print(sum(valid_json) / len(valid_json))
    # print(sum(correct) / len(correct))


if __name__ == "__main__":
    run_eval()
