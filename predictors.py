import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dataset import Sample
from typing import Any, Dict, List


device = "cuda" if torch.cuda.is_available() else "cpu"
# None means we use regular attention
ATTN_TO_USE = "flash_attention_2" if torch.cuda.is_available() else None

# TODO: should we force it to return a json dict?
# TODO: unify generation config
SYS_PROMPT = (
    "You are an AI agent used for automation. Do not act like a chatbot. Execute the task and"
    "follow the instructions for the formatting of the output"
)
# Setting this number rather high to give room for the COT answer and get to the json formatting
# that will happen at the very end
MAX_NEW_TOKENS = 1000


class Predictor:
    @staticmethod
    def format_prompt(sample: Sample) -> str:
        raise NotImplementedError

    def predict(self, sample: Sample) -> str:
        raise NotImplementedError

    def post_process_output(self, outputs: torch.Tensor) -> str:
        raise NotImplementedError

    @property
    def generation_config(self):
        return GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def collate_fn(self, batch: List[Sample]):
        # TODO: issue with declaring tokenizer type
        # To more easily reuse the base tokenizer batch_encode, we make it a predictor method
        # and embed the model-specific formatting in it
        prompts = [self.format_prompt(sample) for sample in batch]
        # Will pad left
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        )
        return {**inputs, 'samples': batch}


class MistralOpenOrcaPredictor(Predictor):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "Open-Orca/Mistral-7B-OpenOrca",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation=ATTN_TO_USE,
        ).to(device)
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
        # the tokenizer doesn't natively have a pad token
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    @staticmethod
    def format_prompt(sample: Sample) -> str:
        prompt = ' '.join([sample.task_input, sample.task_definition])
        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"
        sys_format = prefix + "system\n" + SYS_PROMPT + suffix
        user_format = prefix + "user\n" + prompt + suffix
        assistant_format = prefix + "assistant\n"
        input_text = sys_format + user_format + assistant_format
        return input_text

    def predict(self, batch: Dict[str, Any]):
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"].to("cuda"),
            attention_mask=batch["attention_mask"].to("cuda"),
            generation_config=self.generation_config,
        )
        return self.post_process_output(generated_ids)

    def post_process_output(self, outputs: torch.Tensor) -> str:
        # Assuming the output will not contain "assistant" more than once
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class MistralInstructPredictor(Predictor):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation=ATTN_TO_USE,
        ).to(device)
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2"
        )
        # the tokenizer doesn't natively have a pad token
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    @staticmethod
    def format_prompt(sample: Sample) -> str:
        return " ".join(
            ["[INST]", sample.task_input, sample.task_definition, "[/INST]"]
        )

    def predict(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"].to("cuda"),
            attention_mask=batch["attention_mask"].to("cuda"),
            generation_config=self.generation_config,
        )
        return self.post_process_output(generated_ids)

    def post_process_output(self, outputs: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
