import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams



load_dotenv()
url = os.getenv("QDRANT_ENDPOINT")
collection_name = os.getenv("COLLECTION_NAME")
client = AsyncQdrantClient(url="http://" + url)

async def ensure_collection():
    existing = await client.collection_exists(collection_name)

    if not existing:

        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=int(os.getenv("DIMENSIONS")),
                distance=Distance.COSINE,
            ),
        )