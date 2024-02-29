import json
import os

import click
from transformers import pipeline

from predictors import (
    MistralOpenOrcaPredictor,
    MistralInstructPredictor,
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
    "--model", type=click.Choice(list(MODEL_CHOICES.keys())), default="mistral-instruct",
)
@click.option("--json_mode", is_flag=True)
def run_eval(model: str, json_mode: bool) -> None:
    # model_predictor_cls = MODEL_CHOICES[model]
    model_predictor = MistralInstructPredictor()

    from transformers import pipeline

    pipe = pipeline("text-generation",
                    model=model_predictor.model,
                    tokenizer=model_predictor.tokenizer,
                    batch_size=1,
                    max_new_tokens=512, device=0)
    dataset = OpenMathDataset()
    for i in range(1,10):
        sample = dataset[i]
        # prompt = model_predictor.format_sample_into_prompt(sample)

        # if json_mode:
        from jsonformer import Jsonformer
        json_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "number"},
            }
        }
        print(sample)
        # Jsonformer should take care of the input
        # jsonformer = Jsonformer(model_predictor.model, model_predictor.tokenizer, json_schema, prompt)
        # generated_data = jsonformer()
        #
        # print('with json_mode', generated_data)

        answer = pipe(sample.task_input + 'First reason and find the solution to the question. Then format the answer of the question as a json dictionnary with a key "answer" and the value the corresponding numerical answer to the question. Make sure there is one and only one json dict in your answer')

        print('no json mode', answer)
        print('recovered', find_and_parse_json(answer))

        print('\n')



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
