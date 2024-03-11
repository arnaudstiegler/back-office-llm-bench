import json
import os
from dataclasses import asdict
import click
from jsonformer import Jsonformer
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import OpenMathDataset, KleisterNdaDataset, MultiHopQADataset
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


def run_eval(
    model: str, output_dir: str, batch_size: int, dataset_name: str, json_mode: bool
) -> None:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model_predictor = MODEL_CHOICES[model]()

    if dataset_name == "open_math":
        dataset = OpenMathDataset(json_mode=json_mode)
    elif dataset_name == "kleister_nda":
        dataset = KleisterNdaDataset(json_mode=json_mode)
    elif dataset_name == "multi_hop_qa":
        dataset = MultiHopQADataset(json_mode=json_mode)
    else:
        raise ValueError(f"No such dataset: {dataset_name}")

    predictions = []
    if json_mode:
        # There is no possibility to batch inference with jsonformer (AFAIK)
        for i in tqdm(range(1, len(dataset) + 1)):
            sample = dataset[i]
            json_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    # Has to be a string for OpenMath as the answer can sometimes be an
                    # expression rather than numerical
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

            print(answer, asdict(sample))
            print("\n")

            predictions.append(
                {
                    "sample": asdict(sample),
                    "answer": answer,
                }
            )
            json.dump(
                predictions,
                open(
                    os.path.join(output_dir, f"{model}_json-mode={json_mode}.json"), "w"
                ),
            )

    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=model_predictor.collate_fn,
        )
        for batch in tqdm(train_dataloader):
            preds = model_predictor.predict(batch)

            for pred, sample in zip(preds, batch["samples"]):
                predictions.append(
                    {
                        "sample": asdict(sample),
                        "answer": pred,
                    }
                )

            print(pred, asdict(sample))
            print("\n")

            json.dump(
                predictions,
                open(
                    os.path.join(output_dir, f"{model}_json-mode={json_mode}.json"), "w"
                ),
            )


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
@click.option("--batch_size", type=int, default=1)
@click.option("--json_mode", is_flag=True)
def run_eval_cli(model: str, output_dir: str, batch_size: int, json_mode: bool) -> None:
    return run_eval(model, output_dir, batch_size, json_mode)


if __name__ == "__main__":
    run_eval_cli()
