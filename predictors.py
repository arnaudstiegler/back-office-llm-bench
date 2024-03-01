import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dataset import Sample

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
    def format_sample_into_prompt(sample: Sample) -> str:
        return sample.task_input + " " + sample.task_definition

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


class MistralOpenOrcaPredictor(Predictor):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "Open-Orca/Mistral-7B-OpenOrca",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation=ATTN_TO_USE,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")

    def predict(self, sample: Sample):
        prompt = self.format_sample_into_prompt(sample)
        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"
        sys_format = prefix + "system\n" + SYS_PROMPT + suffix
        user_format = prefix + "user\n" + prompt + suffix
        assistant_format = prefix + "assistant\n"
        input_text = sys_format + user_format + assistant_format

        inputs = self.tokenizer(
            input_text, return_tensors="pt", return_attention_mask=True
        ).to(device)
        outputs = self.model.generate(
            **inputs, generation_config=self.generation_config
        )

        return self.post_process_output(outputs)

    def post_process_output(self, outputs: torch.Tensor) -> str:
        # Assuming the output will not contain "assistant" more than once
        return (
            self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            .split("assistant")[1]
            .strip()
        )


class MistralInstructPredictor(Predictor):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation=ATTN_TO_USE,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2"
        )

    def predict(self, sample: Sample) -> str:
        prompt = self.format_sample_into_prompt(sample)
        encoded_prompt = self.tokenizer(
            "[INST]" + prompt + "[/INST]", return_tensors="pt"
        )["input_ids"]

        generated_ids = self.model.generate(
            encoded_prompt.to("cuda"), generation_config=self.generation_config
        )
        answer = self.post_process_output(generated_ids)
        return answer

    def post_process_output(self, outputs: torch.Tensor) -> str:
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
