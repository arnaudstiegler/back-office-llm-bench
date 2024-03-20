from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
import os
from datetime import datetime
from typing import Dict, Any
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, \
    FullStateDictConfig

dataset = load_dataset("textvqa")
sample = dataset['train'][0]

# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoProcessor.from_pretrained(
    "adept/fuyu-8b"
)
model = AutoModelForCausalLM.from_pretrained(
    "adept/fuyu-8b",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    quantization_config=bnb_config
)
model = torch.compile(model)
