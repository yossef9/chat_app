import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Create client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Send a simple chat message
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you respond to this message?"}
    ],
    max_tokens=50
)

print("AI Response:", response.choices[0].message.content)
