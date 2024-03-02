import json
import os
from dataclasses import asdict

import click
from jsonformer import Jsonformer
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import OpenMathDataset
from predictors import (
    MistralOpenOrcaPredictor,
    MistralInstructPredictor,
    MAX_NEW_TOKENS,
)

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

    if json_mode:
        # There is no possibility to batch inference with jsonformer (AFAIK)
        for i in tqdm(range(1, 1001)):
            sample = dataset[i]
            json_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    # Has to be a string for OpenMath as the answer can sometimes be an expression rather than numerical
                },
            }
            prompt = model_predictor.format_prompt(sample)
            jsonformer = Jsonformer(
                model_predictor.model,
                model_predictor.tokenizer,
                json_schema,
                prompt,
                max_string_token_length=MAX_NEW_TOKENS,
            )
            answer = jsonformer()

            predictions.append(
                {
                    "sample": asdict(sample),
                    "answer": answer,
                }
            )

    else:
        train_dataloader = DataLoader(
            dataset, batch_size=10, shuffle=False, collate_fn=model_predictor.collate_fn
        )
        for batch in tqdm(train_dataloader):
            preds = model_predictor.predict(batch)
            print(preds)
            import ipdb

            ipdb.set_trace()

    json.dump(
        predictions,
        open(os.path.join(output_dir, f"{model}_json-mode={json_mode}.json"), "w"),
    )


if __name__ == "__main__":
    run_eval()
