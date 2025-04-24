from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",
        torch_dtype="auto",
    )

app = FastAPI(lifespan=lifespan)



@app.get("/question")
async def predict(x: float):
    result = ml_models["answer_to_everything"](x)
    return {"result": result}


@app.post("/upload_document")
async def upload_document(document: UploadFile):
    pass