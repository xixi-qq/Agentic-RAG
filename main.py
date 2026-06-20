from contextlib import asynccontextmanager

from fastapi import FastAPI
from apps import user,rag
from config.qdrant_config import client
import os
from dotenv import load_dotenv



load_dotenv()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    exists = await client.collection_exists(COLLECTION_NAME)

    if not exists:
        raise RuntimeError(
            f"Qdrant Collection 不存在: {COLLECTION_NAME}"
        )

    try:
        yield
    finally:
        await client.close()



app = FastAPI(lifespan=lifespan)



app.include_router(user.router.router)
app.include_router(rag.router.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}




##############CORS#################
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],)

# app.add_middleware(AuthMiddleware)



