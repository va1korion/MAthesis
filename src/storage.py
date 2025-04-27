from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing_extensions import List, TypedDict
from preprocessing import embeddings, Chroma, parse_docs, parse_doc
from loguru import logger

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="../chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

def vectorise_dir(directory: str):
    docs = parse_docs(directory)
    logger.info(f"Parsed all docs: {len(docs)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    logger.info(f"Split all docs! Len: {len(all_splits)}")

    _ = vector_store.add_documents(documents=all_splits)
    logger.info(f"Added all docs!")

def vectorise_doc(directory: str):
    parse_doc(directory)


# Define application steps
def retrieve(question: str) -> [Document]:
    retrieved_docs = vector_store.similarity_search(question)
    return retrieved_docs

if __name__ == "__main__":
    vectorise_dir("../example_data")
    print(retrieve("Что изучается в этой дисциплине"))
