import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import find_and_parse_json


ATTN_TO_USE = "flash_attention_2" if torch.cuda.is_available() else None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = AutoModelForCausalLM.from_pretrained(
#             "Open-Orca/Mistral-7B-OpenOrca",
#             torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#             attn_implementation=ATTN_TO_USE,
#         )
# tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    attn_implementation=ATTN_TO_USE,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2"
)
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024, # This should be high to give room to the model
                device=0)

prompt = '''
A school has 15 classrooms. One-third of these classrooms have 30 desks in each classroom and the rest have 25 desks in each classroom. Only one student can sit at one desk. How many students can this school accommodate so that everyone has their own desk?

First reason and find the solution to the question. Then format the answer of the question as a json dictionnary with a key "answer" and the value the corresponding numerical answer to the question. Make sure there is one and only one json dict in your answer
'''
out = pipe("[INST]" + prompt + '[/INST]')

answer = find_and_parse_json(out[0]['generated_text'])
