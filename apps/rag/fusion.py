from sqlalchemy.ext.asyncio import AsyncSession

from apps.rag.bm25 import search_bm25
from apps.rag.reranker import rerank_chunks
from apps.rag.retrieval import retrieve_chunk, deduplicate_by_content
from apps.rag.schemas import RetrieveItem


def merge_candidates(
    *result_lists: list[RetrieveItem],
) -> list[RetrieveItem]:
    merged = {}

    for results in result_lists:
        for item in results:
            key = (
                item.metadata.document_id,
                item.metadata.chunk_index,
            )

            if key not in merged:
                merged[key] = item

    return list(merged.values())



async def retrieve_and_rerank(
    query: str,
    user_id: int,
    document_id: int | None,
    db: AsyncSession,
    candidate_k: int = 30,
    final_k: int = 5,
    score_threshold: float = 0.0,
):
    vector_results = await retrieve_chunk(
        user_query=query,
        user_id=user_id,
        document_id=document_id,
        db=db,
        top_k=candidate_k,
        score_threshold=score_threshold,
    )

    bm25_results = await search_bm25(
        query=query,
        user_id=user_id,
        document_id=document_id,
        db=db,
        top_k=candidate_k,
    )

    candidates = merge_candidates(
        vector_results,
        bm25_results,
    )
    if not candidates:
        return []
    candidates = deduplicate_by_content(candidates)

    return await rerank_chunks(
        query=query,
        items=candidates,
        top_n=final_k,
    )
