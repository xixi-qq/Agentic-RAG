import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from apps.rag.crud import get_chunk_by_vector_id
from apps.rag.embedding import embed_query
from apps.rag.schemas import RetrieveItem, RetrieveItemMetadata
from config.qdrant_config import client
from qdrant_client.models import FieldCondition,MatchValue,Filter
import re


load_dotenv()
collection_name = os.getenv("COLLECTION_NAME")



def deduplicate_chunks(retrieve_results: list[RetrieveItem]) -> list[RetrieveItem]:
    """
    去除重复的chunk
    """
    unique = {}
    for item in retrieve_results:
        key = (item.metadata.document_id, item.metadata.chunk_index)
        existing = unique.get(key)

        if existing is None or item.score > existing.score:
            unique[key] = item
    return sorted(unique.values(),key=lambda item: item.score,reverse=True)




def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_by_content(items):
    unique = {}

    for item in items:
        key = normalize_text(item.content)

        existing = unique.get(key)

        if existing is None or item.score > existing.score:
            unique[key] = item

    return sorted(
        unique.values(),
        key=lambda item: item.score,
        reverse=True,
    )

from difflib import SequenceMatcher


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        normalize_text(left),
        normalize_text(right),
    ).ratio()

def deduplicate_similar(items, threshold: float = 0.9):
    sorted_items = sorted(
        items,
        key=lambda item: item.score,
        reverse=True,
    )

    selected = []

    for candidate in sorted_items:
        duplicate = any(
            text_similarity(candidate.content, item.content) >= threshold
            for item in selected
        )

        if not duplicate:
            selected.append(candidate)

    return selected

async def search_vector(query: str,user_id: int,document_id: int | None, score_threshold: float,top_k=5):
    query_vector = await embed_query(query)
    conditions = [
        FieldCondition(key="user_id",match=MatchValue(value=user_id),
        ),
    ]
    if document_id:
        conditions.append(
            FieldCondition(key="document_id",match=MatchValue(value=document_id),
            )
        )

    response = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=Filter(must=conditions),
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,
    )
    return response.points


async def retrieve_chunk(
        user_query: str,
        user_id: int,
        document_id: int | None,
        db: AsyncSession,
        top_k: int = 15,
        score_threshold: float = 0.5) -> list[RetrieveItem]:

    points = await search_vector(
        user_query,
        user_id,
        document_id,
        score_threshold,
        top_k,
    )
    if not points:
        return []
    results = []
    vector_ids = [str(point.id) for point in points]
    chunk_rows = await get_chunk_by_vector_id(user_id,vector_ids, db)
    chunks_dic = {
        chunk.vector_id: (chunk, filename)
        for chunk, filename in chunk_rows
    }
    for point in points:
        chunk_data = chunks_dic.get(str(point.id))
        if not chunk_data:
            continue
        chunk, filename = chunk_data
        results.append(RetrieveItem(content=chunk.content,
                                    score=point.score,
                                    metadata=RetrieveItemMetadata(document_id=chunk.document_id,
                                                                  page_number=chunk.page_number,
                                                                  filename=filename,
                                                                  chunk_index=chunk.chunk_index)))

    return results






