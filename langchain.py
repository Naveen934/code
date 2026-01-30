from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain_core.runnables import RunnablePassthrough

import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub

# Set your HuggingFace API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.1,
        max_length=512,  # Maximum tokens to generate
        top_p=0.95,  # Nucleus sampling parameter
        repetition_penalty=1.15,  # Reduce repetition
        do_sample=True,  # Enable sampling
        return_full_text=False,  # Return only generated text
        # Additional parameters for better control
        model_kwargs={
            "temperature": 0.1,
            "max_new_tokens": 512,
            "top_k": 50
        }
    )

# Assuming you have examples defined somewhere
# examples = [
#     {"query": "What is 2+2?", "answer": "2+2 equals 4"},
#     {"query": "What is the capital of France?", "answer": "The capital of France is Paris"},
#     # ... more examples
# ]

# Define your examples (make sure this is defined)
examples = [
    {"query": "What is 2+2?", "answer": "2+2 equals 4"},
    {"query": "What is the capital of France?", "answer": "The capital of France is Paris"},
    # Add more examples here
]

# 1. Few-Shot Prompt Template with Example Selector
example_template = """
Question: {query}
Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """You are a {template_ageoption}, and {template_tasktype_option}: 
Here are some examples: 
"""

suffix = """
Question: {template_userInput}
Response: """

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=200
)

few_shot_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["template_userInput", "template_ageoption", "template_tasktype_option"],
    example_separator="\n"
)

# Create a chain with proper invoke pattern
from langchain_core.runnables import RunnableSequence

# Method 1: Direct invoke with formatted prompt
formatted_prompt = few_shot_prompt_template.format(
    template_userInput=query,
    template_ageoption=age_option,
    template_tasktype_option=tasktype_option
)

# Using invoke method
response = llm.invoke(formatted_prompt)
print(response.content)  # For Chat models
# or print(response) for LLM models

# Method 2: Using Runnable sequence (recommended for complex chains)
chain = few_shot_prompt_template | llm
response = chain.invoke({
    "template_userInput": query,
    "template_ageoption": age_option,
    "template_tasktype_option": tasktype_option
})
print(response.content)

# 2. Pandas DataFrame Agent (updated syntax)
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor

# Create the agent
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    # Additional parameters for better control
    agent_type="openai-tools",  # or "zero-shot-react-description"
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate"
)

# Using invoke method
try:
    response = agent.invoke({"input": query})
    print(response["output"])
except Exception as e:
    print(f"Error: {e}")

# Alternative: Using AgentExecutor explicitly
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent,
    tools=agent.tools,
    verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke({"input": query})
print(response["output"])

# 3. DuckDuckGo Search Tool (updated import)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

# Create search tool
search_tool = DuckDuckGoSearchRun()

# Use it directly
search_result = search_tool.invoke("Obama's first name?")
print(search_result)

# Or create a tool for use with agents
search_tool = Tool(
    name="DuckDuckGo Search",
    func=search_tool.invoke,
    description="Useful for searching the internet for current information"
)

# Example of using with an agent
from langchain.agents import initialize_agent, AgentType

tools = [search_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

response = agent.invoke({"input": "What's the latest news about AI?"})
print(response["output"])