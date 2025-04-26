from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from src.generate import Generator
from src.storage import vectorise_dir

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    Gen = Generator()
    vectorise_dir("../data")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/question")
async def predict():
    result = ml_models["answer_to_everything"]
    return {"result": result}


@app.post("/upload_document")
async def upload_document(document: UploadFile):
    pass