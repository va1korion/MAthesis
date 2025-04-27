from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles
from generate import generator
from storage import vectorise_dir, retrieve
from loguru import logger

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Starting lifespan")
    vectorise_dir("../example_data")
    logger.info("App is running")
    yield
    logger.info("App has been stopped")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="front"), name="static")


@app.get("/question")
async def predict(question: str):
    context = retrieve(question)
    result = generator.generate(question=question, context=context)
    return {"result": result}


class State(BaseModel):
    question: str
    context: str
    system_prompt: str

@app.get("/question_with_ctx")
async def predict(state: State):
    result = generator.generate_plain(question=state.question, context=state.context, system_prompt=state.system_prompt)
    return {"result": result}

@app.post("/upload_document")
async def upload_document(document: UploadFile):
    pass