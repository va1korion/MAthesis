import asyncio
import time
from langchain.chains import llm
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from pydantic import BaseModel
from storage import vectorise_dir, retrieve
from loguru import logger
import httpx
import logging
from typing import List
import os

class LLMClient:
    def __init__(self, server_url: str = os.getenv("LLAMA_SERVER_URL", "http://llama-server:8000")):
        self.server_url = server_url

    async def generate(self, context: List[Document], question: str) -> str:
        docs_content = "\n\n".join(doc.page_content for doc in context)

        prompt = (
            "Ты ассистент студенческого офиса Университета. "
            "Ты должен отвечать на вопросы исходя из контекста, представленного ниже. "
            "Если контекст не содержит ответа на вопрос, ответь, что данных нет, не пытайся придумать ответ и не выходи за рамки контекста. "
            f"\n\nКонтекст:\n{docs_content}\n\nВопрос: {question}\nОтвет:"
        )

        payload = {
            "prompt": prompt,
            "n_predict": 16384,
            "stream": False
        }

        generation_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.server_url}/completion", json=payload, timeout=100)
            response.raise_for_status()
            result = response.json()

        elapsed = time.time() - generation_time
        generated_text = result.get("content", "Данных нет").strip()
        logger.info(f"Generated {len(generated_text.split())} words in {elapsed:.2f} seconds")
        return generated_text


generator = LLMClient()

if __name__ == "__main__":
    # vectorise_dir("../example_data")
    context = retrieve("Что изучается в этой дисциплине: Системное программное обеспечение")
    print(asyncio.run(generator.generate(context, question="Что изучается в этой дисциплине: Системное программное обеспечение?")))




