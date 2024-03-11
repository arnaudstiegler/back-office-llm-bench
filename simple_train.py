import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
import os
from datetime import datetime
from typing import Dict, Any


MAX_STEPS = 1000


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = load_dataset("Open-Orca/OpenOrca", split="train[:95%]")
    val = load_dataset("Open-Orca/OpenOrca", split="train[99%:]")

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    ).to(device)
    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        truncation=True,
        model_max_length=512,
        padding="max_length",
    )
    # the tokenizer doesn't natively have a pad token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    model.gradient_checkpointing_enable()

    def prepare_sample(sample: Dict[str, str]) -> Dict[str, Any]:
        question = tokenizer(
            sample["question"] + sample["response"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        # Note: supervising the question as well, might be changed later
        return {
            "input_ids": question["input_ids"],
            "attention_mask": question["attention_mask"],
            "labels": question["input_ids"].clone(),
        }

    preprocessed_val_map = val.map(prepare_sample)

    # model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    # print_trainable_parameters(model)
    #
    # # Apply the accelerator. You can comment this out to remove the accelerator.
    # model = accelerator.prepare_model(model)

    # wandb.login()
    #
    # wandb_project = "test-orca"
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    #
    project = "test-open-orca"
    base_model_name = "mistral"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    preprocessed_val_map = preprocessed_val_map.remove_columns(
        ["id", "system_prompt", "question", "response"]
    )

    trainer = Trainer(
        model=model,
        train_dataset=preprocessed_val_map,  # TODO: update
        eval_dataset=preprocessed_val_map,
        args=TrainingArguments(
            output_dir="test_run/",
            warmup_steps=int(0.1 * MAX_STEPS),  # TODO: update that
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            remove_unused_columns=False,
            max_steps=MAX_STEPS,
            learning_rate=2.5e-5,  # Want about 10x smaller than the Mistral learning rate
            logging_steps=50,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=50,  # Save checkpoints every 50 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=50,  # Evaluate and save checkpoints every 50 steps
            do_eval=True,  # Perform evaluation at the end of training
            # report_to="wandb",           # Comment this out if you don't want to use weights & baises
            # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()


if __name__ == "__main__":
    train()
