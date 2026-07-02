import os

from dotenv import load_dotenv
from redis import Redis

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_conn = Redis.from_url(
    REDIS_URL,
    decode_responses=False,
)
