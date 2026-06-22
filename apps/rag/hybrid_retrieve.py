from sqlalchemy.ext.asyncio import AsyncSession

from apps.rag.bm25 import search_bm25
from apps.rag.retrieval import retrieve_chunk
from apps.rag.schemas import RetrieveItem


def make_chunk_key(item: RetrieveItem) -> tuple[int, int]:
    """同一文档中的chunk唯一标识。"""
    return (
        item.metadata.document_id,
        item.metadata.chunk_index,
    )



def reciprocal_rank_fusion(result_lists: list[list[RetrieveItem]],rrf_k: int,top_k: int) -> list[RetrieveItem]:

    fused_scores: dict[tuple[int, int], float] = {}
    item_mapping: dict[tuple[int, int], RetrieveItem] = {}
    for results in result_lists:
        for rank,item in enumerate(results,start=1):
            key = make_chunk_key(item)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in item_mapping:
                item_mapping[key] = item
    ranked_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True,
    )
    fused_results = []
    for key in ranked_keys[:top_k]:
        item = item_mapping[key]

        fused_results.append(
            item.model_copy(
                update={
                    "score": fused_scores[key],
                }
            )
        )

    return fused_results




async def hybrid_retrieve(
    user_query: str,
    user_id: int,
    document_id: int | None,
    db: AsyncSession,
    candidate_k: int = 30,
    final_k: int = 30,
    score_threshold: float = 0.0,
    rrf_k: int = 60,
) -> list[RetrieveItem]:
    vector_results = await retrieve_chunk(
        user_query=user_query,
        user_id=user_id,
        document_id=document_id,
        db=db,
        top_k=candidate_k,
        score_threshold=score_threshold,
    )

    bm25_results = await search_bm25(
        query=user_query,
        user_id=user_id,
        document_id=document_id,
        db=db,
        top_k=candidate_k,
    )
    return reciprocal_rank_fusion(
        result_lists=[
            vector_results,
            bm25_results,
        ],
        rrf_k=rrf_k,
        top_k=final_k,
    )

