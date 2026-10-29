import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
print("--- Starting Script ---")
load_dotenv()
api_key = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

print(f"Loaded API Key Status: {'Found Key (Starts with ' + api_key[:7] + ')' if api_key else 'Not Found (None)'}")

# Initialize model

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.6)
    
ans = llm.invoke("I want to know how to make money after 60")
print(ans.content)