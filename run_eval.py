import json
import os

import click
from jsonformer import Jsonformer

from dataset import OpenMathDataset
from predictors import (
    MistralOpenOrcaPredictor,
    MistralInstructPredictor,
    MAX_NEW_TOKENS,
)
from dataclasses import asdict
from tqdm import tqdm

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
@click.option(
    "--output_dir",
    type=str,
    default="/home/ubuntu/predictions/",
)
@click.option("--json_mode", is_flag=True)
def run_eval(model: str, output_dir: str, json_mode: bool) -> None:
    model_predictor = MistralInstructPredictor()
    dataset = OpenMathDataset()

    predictions = []
    for i in tqdm(range(1, 1001)):
        sample = dataset[i]

        if json_mode:
            json_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},  # Has to be a string for OpenMath as the answer can sometimes be an expression rather than numerical
                },
            }
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
            answer = jsonformer()

        else:
            answer = model_predictor.predict(sample)

        predictions.append(
            {
                'sample': asdict(sample),
                'answer': answer,
            }
        )

        json.dump(predictions, open(os.path.join(output_dir, f'{model}_json-mode={json_mode}.json'), 'w'))


if __name__ == "__main__":
    run_eval()
