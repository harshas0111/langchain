import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
api_key = os.getenv("HF_API_KEY")
if not api_key:
    raise ValueError(
        "API key not found. Check OPENAI_KEY in your .env file."
    )

client = InferenceClient(
    

   api_key=api_key,
   
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b:fastest",
    messages=[
        {
            "roles": "user",
             "content": "what is AI"
        }
    ],
    temperature=0.6,
    max_tokens=200,
)
print(response.choices[0].content)