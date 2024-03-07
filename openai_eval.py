from openai import OpenAI
from openai.types.chat import ChatCompletion
from dataset import OpenMathDataset
from utils import find_and_parse_json
import json
from tqdm import tqdm

SYS_PROMPT = (
    "You are an AI agent used for automation. Do not act like a chatbot. Execute the task and"
    "follow the instructions for the formatting of the output as a JSON object."
)
INPUT_TOKEN_COST = 1e-5
OUTPUT_TOKEN_COST = 3e-5

client = OpenAI()
dataset = OpenMathDataset(json_mode=False)


def compute_cost(response: ChatCompletion):
    return INPUT_TOKEN_COST*response.usage.prompt_tokens + OUTPUT_TOKEN_COST*response.usage.completion_tokens

predictions = []
for i in tqdm(range(1, 1001)):
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": dataset[i].prompt + ' Think step by step'},
      ],
        response_format={ "type": "json_object" }
    )
    predictions.append(response.dict())
    json.dump(predictions, open('predictions_json_mode.json', 'w'))

# responses = json.load(open('predictions_not_json_mode.json'))
# cost = []
#
# for response in responses:
#     cost.append(compute_cost(ChatCompletion(**response)))
#
# print(f'Average cost: {sum(cost) / len(cost)}$')
# print(f'Cost for 1k samples: {(sum(cost) / len(cost))*1000}$')