from langchain.chains import llm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from src.storage import prompt, State


class Generator:
    MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",
        torch_dtype="auto",
    )


    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})

        messages = [{"role": "user", "content": "Для чего нужна токенизация?"}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(input_ids, max_new_tokens=1024)
        print(self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True))
        response = llm.invoke(messages)
        return {"answer": response.content}
