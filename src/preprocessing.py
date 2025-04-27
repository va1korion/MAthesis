import os
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader
from langchain_core.documents import Document
from typing import List
from loguru import logger
from langchain_chroma import Chroma


def parse_doc(filepath) -> [Document]:
    loader = PDFMinerLoader(
        file_path=filepath,
        mode="single",
    )
    return loader.load()


def parse_docs(directory: str) -> List[Document]:
    result = []
    exceptions = set()
    for filename in os.listdir(directory):
        logger.info(f"Parsing {filename}")
        if filename.endswith(".pdf"):
            try:
                result += parse_doc(f"{directory}/{filename}")
            except Exception as e:
                logger.error(f"Failed to parse {filename}: {e}")
                exceptions.add(e)
    logger.info(f"Got {len(exceptions)} exceptions: {exceptions}")
    return result


class USEREmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("deepvk/USER-bge-m3")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()


embeddings=USEREmbeddings()


if __name__ == "__main__":
    print(parse_doc("../data/630.pdf"))
