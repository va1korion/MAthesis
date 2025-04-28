import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles
from generate import generator
from storage import vectorise_dir, retrieve
from loguru import logger
from fastapi.security import OAuth2PasswordBearer
import hmac



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
app.mount("/downloads", StaticFiles(directory="../data"), name="downloads")



@app.get("/question")
async def predict(question: str):
    context = retrieve(question)
    result = await generator.generate(question=question, context=context)
    return {"result": result}


class State(BaseModel):
    question: str
    context: str
    system_prompt: str

@app.get("/question_with_ctx")
async def predict(state: State):
    result = await generator.generate_plain(question=state.question, context=state.context, system_prompt=state.system_prompt)
    return {"result": result}

@app.post("/upload_document")
async def upload_document(document: UploadFile):
    pass


# Configuration class using environment variables
class Settings(BaseModel):
    admin_username: str = os.getenv("APP_ADMIN_USER")
    admin_password: str = os.getenv("APP_ADMIN_PASSWORD")

    class Config:
        env_file = ".env"


security = HTTPBasic()
settings = Settings()


# Verify credentials securely with timing-attack resistant comparison
def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username_match = hmac.compare_digest(credentials.username.encode('utf-8'),
                                         settings.admin_username.encode('utf-8'))
    password_match = hmac.compare_digest(credentials.password.encode('utf-8'),
                                         settings.admin_password.encode('utf-8'))
    if not (username_match and password_match):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.post("/upload/")
async def upload_file(
        username: str = Depends(verify_admin),
        file: UploadFile = File(...)
):
    # Validate file contents here
    try:
        contents = await file.read()
        # Process file contents
        with open(f"../data/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "filename": file.filename,
            "size": len(contents),
            "uploader": username
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        await file.close()