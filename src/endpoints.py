import os
import shutil
from contextlib import asynccontextmanager
from http.client import HTTPResponse
from fastapi import FastAPI, UploadFile, Request
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles
from generate import generator
from storage import vectorise_dir, retrieve
from loguru import logger
from fastapi.security import OAuth2PasswordBearer
import hmac
from starlette.responses import RedirectResponse
import json
from fastapi import FastAPI
from starlette.config import Config
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import HTMLResponse, RedirectResponse
from authlib.integrations.starlette_client import OAuth, OAuthError




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Starting lifespan")
    vectorise_dir("../data")
    logger.info("App is running")
    yield
    logger.info("App has been stopped")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="front"), name="static")
app.mount("/downloads", StaticFiles(directory="../data"), name="downloads")
app.add_middleware(SessionMiddleware, secret_key="!secret")

try:
    logger.info('Trying to register ITMO.ID')
    config = Config('.env')
    oauth = OAuth(config)

    CONF_URL = 'https://id.itmo.ru/auth/realms/itmo/.well-known/openid-configuration'
    oauth.register(
        name='itmo',
        server_metadata_url=CONF_URL,
        client_id=os.getenv('ITMO_CLIENT_ID'),
        client_secret=os.getenv('ITMO_CLIENT_SECRET'),
        client_kwargs={
            'scope': 'openid email profile'
        }
    )
except Exception as e:
    logger.warning(f'Oauth registration failed: {e} \n'
                   'OAuth 2.0 required')

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

@app.get('/auth')
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as error:
        return HTMLResponse(f'<h1>{error.error}</h1>')
    user = token.get('userinfo')
    if user:
        request.session['user'] = dict(user)
    return RedirectResponse(url='/')


@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Validate file contents here
    user = request.session.get('user')
    if not user:
        data = json.dumps(user)
        return RedirectResponse(url='/auth')

    try:
        contents = await file.read()
        # Process file contents
        with open(f"../data/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "filename": file.filename,
            "size": len(contents),
            "uploader": user.get('username', "authenticated"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        await file.close()


@app.delete('/delete/{document_id}')
async def upload_file(request: Request, document_id: str):
    user = request.session.get('user')
    if not user:
        data = json.dumps(user)
        return RedirectResponse(url='/auth')
