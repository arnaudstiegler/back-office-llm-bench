# back-office-llm-bench
Benchmarking LLMs on plain back-office tasks

## TODOS:
- For the FinQA dataset, find the original dataset and/or iterate on the prompt to focus on the last question only
- ~~Add support for OpenMath~~
- ~~Simpler eval pipeline~~
- ~~Retrieve same output as HuggingFaceChat~~
- ~~Find the highest batch size we can handle on a g5~~
- ~~Retrieve the same output as pipeline with a normal generate (to handle jsonformer)~~
- Add batching to the pipeline
- Test out the pipeline stuff with OpenOrca
- Create a metric class to count recovered json and correct answer
- Find easier dataset than OpenMAth

## Goals

- When it comes to executing a task + formatting the answer in a machine-digestible way, how do model perform?
- Chatbot (instruct models) tend to be overly verbose, can that deteriorate results?
- Should you force JSON format (jsonformer or other)? Maybe messes up with Chain-of-thought


## Task types
- Classification (with predefined classes)
  - email classification
- Extract info
  - Extract a list of names from input
  - Find all PII in a document
- Normalize/Reformat an input
  - Translate a city name from one language to another
  - normalize a date to a different format
  - normalize an amount to a different format
  - normalize an ID using an arbitrary rule
- Basic reasoning (what does it mean?):
  - Does an input respect a specific constraint (format for instance)

## Errors that mess up JSON output:

Comments within the json:
'''
{
"revenue": 20000000.0,
"net_profit_margin": 0.15, // 15% as a decimal
"operating_expenses": 5000000.0,
"operating_profit": 3000000.0,
"customer_base_growth": 0.1 // 10% as a decimal
}
'''

Spurious keys in the formating:

'''
{
  "highest_expense": {
    "category": "Groceries",
    "amount": 600
  }
}
'''

instead of:

'''
{
    "category": "Groceries",
    "amount": 600
  }
'''