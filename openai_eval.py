from openai import OpenAI
from openai.types.chat import ChatCompletion
from dataset import MultiHopQADataset
from utils import find_and_parse_json
import json
from tqdm import tqdm

SYS_PROMPT = (
    "You are an AI agent used for automation. Do not act like a chatbot. Execute the task and"
    "follow the instructions for the formatting of the output as a JSON object."
)
INPUT_TOKEN_COST = 1e-5
OUTPUT_TOKEN_COST = 3e-5


def compute_cost(response: ChatCompletion) -> float:
    return (
        INPUT_TOKEN_COST * response.usage.prompt_tokens
        + OUTPUT_TOKEN_COST * response.usage.completion_tokens
    )


client = OpenAI()
json_mode = False
dataset = MultiHopQADataset(json_mode=json_mode)
predictions = []
for i in tqdm(range(len(dataset))):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": dataset[i].prompt},
            ],
            response_format={"type": "json_object"} if json_mode else None,
        )
        response_dict = response.dict()
        from dataclasses import asdict
        response_dict['sample'] = asdict(dataset[i])
        predictions.append(response_dict)
        json.dump(
            predictions,
            open(
                f'openai_predictions/multi_hop_qa/predictions_{"not_" if not json_mode else ""}json_mode.json',
                "w",
            ),
        )
    except Exception as e:
        print(e)

# responses = json.load(open('predictions_not_json_mode.json'))
# cost = []
#
# for response in responses:
#     cost.append(compute_cost(ChatCompletion(**response)))
#
# print(f'Average cost: {sum(cost) / len(cost)}$')
# print(f'Cost for 1k samples: {(sum(cost) / len(cost))*1000}$')
