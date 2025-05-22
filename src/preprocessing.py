import os
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader
from langchain_core.documents import Document
from typing import List
from loguru import logger
from langchain_chroma import Chroma
from pdf2image import convert_from_path
import pytesseract

def parse_doc(filepath) -> [Document]:
    # todo check tesseract
    loader = PDFMinerLoader(
        file_path=filepath,
        mode="single",
    )
    docs = loader.load()

    text = "\n".join(doc.page_content for doc in docs).strip()
    if len(text) > 200:  # adjustable threshold
        return docs

    ocr_texts = []
    images = convert_from_path(filepath)
    for i, img in enumerate(images):
        page_text = pytesseract.image_to_string(img, lang="rus")
        ocr_texts.append(Document(page_content=page_text, metadata={"page": i + 1}))

    return ocr_texts


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
        # should pre-load it, but  it's not too big to download
        self.model = SentenceTransformer("deepvk/USER-bge-m3")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()


embeddings=USEREmbeddings()


if __name__ == "__main__":
    docs = parse_doc("../data/[1403] Параллельное программирование.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    print(len(all_splits))
