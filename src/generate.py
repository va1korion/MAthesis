import asyncio
import time
from langchain.chains import llm
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from pydantic import BaseModel
from storage import vectorise_dir, retrieve
from loguru import logger

class Generator:
    MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",
        torch_dtype="auto",
    )


    async def generate(self, context: list[Document], question: str) -> str:
        docs_content = "\n\n".join(doc.page_content for doc in context)
        generation_time = time.time()
        messages = [{"role": "user", "content": f"Ты ассистент студенческого офиса Университета. Ты должен отвечать на вопросы исходя из контекста, представленного ниже. Если контекст не содержит ответа на вопрос, ответь, что данных нет, не пытайся придумать ответ и не выходи за рамки контекста. "
                                                f"Контекст: {docs_content}"
                                                f"Вопрос: {question} Ответ:"}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(input_ids, max_new_tokens=4096)
        generation_time = time.time() - generation_time
        logger.info(f"Generated {outputs.shape[1] - input_ids.shape[1]} tokens in {generation_time} seconds")
        return self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)


    async def generate_plain(self, question: str, context: str, system_prompt: str) -> str:
        messages = [{"role": "user",
                     "content": f"{system_prompt}"
                                f"Контекст: {context}"
                                f"Вопрос: {question}"}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(input_ids, max_new_tokens=4096)
        return self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)


generator = Generator()

if __name__ == "__main__":
    # vectorise_dir("../example_data")
    context = retrieve("Что изучается в этой дисциплине: Системное программное обеспечение")
    print(asyncio.run(generator.generate(context, question="Что изучается в этой дисциплине: Системное программное обеспечение?")))




