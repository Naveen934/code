from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


prompt_template = """
    Answer the question as detailed as possible from the provided context,\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)






