from openai import OpenAI
client = OpenAI()

# Upload a file with an "assistants" purpose
file = client.files.create(
  file=open("tasks_manual.json", "rb"),
  purpose='assistants'
)

assistant = client.beta.assistants.create(
  instructions="You are helping generating data for a back-office benchmark.",
  model="gpt-4-turbo-preview",
  tools=[{"type": "code_interpreter"}])

thread = client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": """
      Here's a list of tasks. Generate one more sample with the following constraint:
- same type as the ones already in the document
- answer should be unambiguous and found in the input
- output format can be plain text or json""",
      "file_ids": [file.id]
    }
  ]
)
out = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)
import ipdb; ipdb.set_trace()