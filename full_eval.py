from run_eval import run_eval
from datetime import datetime


def format_timestamp(timestamp: datetime) -> str:
    # Format the timestamp into the desired string format
    return timestamp.strftime("%Y-%m-%d_%H:%M:%S")


# Example usage
current_timestamp = datetime.now()
formatted_timestamp = format_timestamp(current_timestamp)
output_dir = "/home/ubuntu/predictions"

run_eval(
    model="mistral-instruct", output_dir=output_dir, batch_size=5, json_mode=False
)
run_eval(model="mistral-instruct", output_dir=output_dir, batch_size=1, json_mode=True)
run_eval(model="mistral-orca", output_dir=output_dir, batch_size=5, json_mode=False)
run_eval(model="mistral-orca", output_dir=output_dir, batch_size=1, json_mode=True)
