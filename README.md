# back-office-llm-bench
Benchmarking LLMs on plain back-office tasks

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