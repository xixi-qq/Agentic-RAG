import os
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")

minio_client = Minio(
    endpoint=os.environ["MINIO_ENDPOINT"],
    access_key=os.environ["MINIO_ACCESS_KEY"],
    secret_key=os.environ["MINIO_SECRET_KEY"],
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
)


def ensure_bucket() -> None:
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)