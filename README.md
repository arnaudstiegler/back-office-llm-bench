# back-office-llm-bench
Benchmarking LLMs on plain back-office tasks


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