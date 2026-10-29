from langchain_ollama import ChatOllama

print("--- Running Local Open Source LLM ---")

# Initialize the locally running Llama model (No API key needed!)
llm = ChatOllama(model="llama3.2", temperature=0.6)

# Invoke the model exactly the same way
ans = llm.invoke("I want to know how to make money after 60")

print(ans.content)
