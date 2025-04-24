import os
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PDFMinerLoader
from langchain_core.documents import Document
from typing import List
from loguru import logger
from langchain_chroma import Chroma

def parse_docs(directory: str) -> List[Document]:
    result = []

    for filename in os.listdir(directory):
        logger.info(f"Parsing {filename}")
        if filename.endswith(".pdf"):
            loader = PDFMinerLoader(
                file_path=f"{directory}/{filename}",
                # headers = None
                # password = None,
                mode="single",
            )
            for chunk in loader.load():
                result.append(chunk)

    return result


def vectorise_dir(directory: str):
    docs = parse_docs(directory)
    vector_store.add_documents(documents=docs)



class USEREmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("deepvk/USER-bge-m3")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query]).tolist()


embeddings=USEREmbeddings()

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="../chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

if __name__ == "__main__":
    print(parse_docs("../data"))
