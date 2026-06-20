import os

from dotenv import load_dotenv
from qdrant_client.http.models import FilterSelector, Filter, FieldCondition, MatchValue
from config.qdrant_config import client



load_dotenv()
collection_name = os.getenv("COLLECTION_NAME")

async def upsert_chunks(vectors):
    points = []
    for vector in vectors:
        points.append(
            {
                "id": vector.vector_id,
                "vector": vector.vector,
                "payload": {
                    "user_id": vector.user_id,
                    "document_id": vector.document_id,
                    "page_number": vector.page_number,
                    "chunk_index": vector.chunk_index,
                    "filename":vector.filename
                },
            }
        )
    await client.upsert(collection_name=collection_name, points=points)

async def delete_vectors(document_id):
    await client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=Filter(must= [FieldCondition(key= "document_id", match= MatchValue(value= document_id),)])))