from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

# 1. Initialize the local pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.6,
        "max_new_tokens": 50
    }
)

# 2. Define a prompt template with variables
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short, one-sentence introduction about {topic}:"
)

# 3. Format the prompt and invoke the model
formatted_prompt = prompt_template.format(topic="Quantum Computing")
response = llm.invoke(formatted_prompt)

print("Prompt:\n", formatted_prompt)
print("\nGenerated Response:\n", response)