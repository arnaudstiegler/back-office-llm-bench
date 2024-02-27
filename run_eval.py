import json
import os
import re

import click

from predictors import (
    MistralOpenOrcaPredictor,
    MistralInstructPredictor,
    Sample,
)

dir_path = os.path.dirname(os.path.realpath(__file__))
samples = json.load(open(os.path.join(dir_path, "tasks_manual.json")))["samples"]

MODEL_CHOICES = {
    "mistral-orca": MistralOpenOrcaPredictor,
    "mistral-instruct": MistralInstructPredictor,
}


def find_and_parse_json(s):
    # Regular expression pattern to find a JSON object
    # This pattern assumes the JSON does not contain nested objects or arrays
    pattern = r"\{[^{}]*\}"

    # Search for JSON string within the input
    match = re.search(pattern, s)

    # If a match is found, parse the JSON string
    if match:
        json_str = match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            return None
    return None


@click.command()
@click.option(
    "--model", type=click.Choice(list(MODEL_CHOICES.keys())), default="mistral-instruct"
)
def run_eval(model: str, json_mode = True) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    if json_mode:
        from jsonformer import Jsonformer
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "is_student": {"type": "boolean"},
                "courses": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }

        prompt = "Generate a person's information based on the following schema:"
        import ipdb; ipdb.set_trace()
        jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
        generated_data = jsonformer()

        print(generated_data)

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
