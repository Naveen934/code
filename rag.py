#choose llm model


#Get a Data Loader / Data Injection
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tamil_Nadu")
data = loader.load()
data


from langchain_community.document_loaders import PyPDFDirectoryLoader
loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
docs=loader.load()

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


with open('kb.text', 'r') as f:
    print(f.read())
context = open('kb.text', 'r').read()

documents = [
    "Large Language Models (LLMs) are transforming AI.",
    "FAISS is a powerful vector database for search.",
    "Retrieval-Augmented Generation (RAG) enhances LLM responses.",
    "Vector embeddings represent text numerically.",
    "LangChain makes working with LLMs easier.",
]

# Create Document objects
docs = [Document(page_content=text) for text in documents]






# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()
document = text_splitter.split_documents(data)


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)


text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
final_documents=text_splitter.split_documents(docs[:20]) #splitting




# Store splits
from langchain_objectbox.vectorstores import ObjectBox
vector = ObjectBox.from_documents(document, GoogleGenerativeAIEmbeddings(model = "models/embedding-001"), embedding_dimensions=768)


from langchain_community.vectorstores import FAISS
vectors=FAISS.from_documents(final_documents,embedding=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")) 


from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GoogleGenerativeAIEmbeddings(model = "models/embedding-001"))




# RAG prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")




# RetrievalQA
from langchain.chains import RetrievalQA
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )


qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)


#result
question = "Explain Prehistory (before 5th century BCE) of tamil nadu"
result = qa_chain({"query": question })
result




# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


loader = TextLoader("../../modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)