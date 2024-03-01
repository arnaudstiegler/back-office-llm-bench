import json
import re
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def find_and_parse_json(s):
    # Regular expression pattern to find a JSON object
    # This pattern assumes the JSON does not contain nested objects or arrays
    pattern = r"\{[^{}]*\}"

    # Search for JSON string within the input
    match = re.search(pattern, s)

    # If a match is found, parse the JSON string
    if match:
        json_str = match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            logger.warning(f"Encountered Json decoder error with input: {json_str}")
            return None
    logger.warning(f"Regex did not match anything with input: {s}")
    return None
