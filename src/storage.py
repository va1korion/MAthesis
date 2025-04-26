from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing_extensions import List, TypedDict
from preprocessing import embeddings, Chroma, parse_docs

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="../chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

def vectorise_dir(directory: str):
    docs = parse_docs(directory)
    vector_store.add_documents(documents=docs)


# Load and chunk contents of the blog
loader = PyPDFLoader()
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(BaseModel):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

