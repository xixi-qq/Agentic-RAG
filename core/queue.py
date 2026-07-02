import os
from dotenv import load_dotenv
from rq import Queue

from config.redis_config import redis_conn

load_dotenv()

RQ_DEFAULT_QUEUE = os.getenv("RQ_DEFAULT_QUEUE", "default")
RQ_RAG_QUEUE = os.getenv("RQ_RAG_QUEUE", "rag")

redis_cli = redis_conn

default_queue = Queue(
    RQ_DEFAULT_QUEUE,
    connection=redis_cli,
)

rag_queue = Queue(
    RQ_RAG_QUEUE,
    connection=redis_cli,
)