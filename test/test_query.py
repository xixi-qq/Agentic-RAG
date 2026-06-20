from unittest.mock import AsyncMock

import pytest

from apps.rag import router
from apps.rag.schemas import (
    QueryRequest,
    QueryResponse,
    RetrieveItem,
    RetrieveItemMetadata,
)


def make_retrieve_item() -> RetrieveItem:
    return RetrieveItem(
        content="测试检索内容",
        score=0.88,
        metadata=RetrieveItemMetadata(
            document_id=9,
            page_number=2,
            filename="test.pdf",
            chunk_index=4,
        ),
    )


def make_item(
    content: str,
    score: float,
    document_id: int,
    chunk_index: int,
) -> RetrieveItem:
    return RetrieveItem(
        content=content,
        score=score,
        metadata=RetrieveItemMetadata(
            document_id=document_id,
            page_number=1,
            filename=f"{document_id}.txt",
            chunk_index=chunk_index,
        ),
    )


async def test_query_returns_fallback_without_calling_model(monkeypatch):
    retrieve_chunk = AsyncMock(return_value=[])
    rerank_chunks = AsyncMock()
    response_user_query = AsyncMock()
    db = object()

    monkeypatch.setattr(router, "retrieve_chunk", retrieve_chunk)
    monkeypatch.setattr(router, "rerank_chunks", rerank_chunks)
    monkeypatch.setattr(router, "response_user_query", response_user_query)

    response = await router.query(
        request=QueryRequest(
            user_query="没有答案的问题",
            document_id=None,
            top_k=7,
            score_threshold=0.66,
        ),
        user_info={"user_id": 3},
        db=db,
    )

    assert isinstance(response, QueryResponse)
    assert response.answer == "未在文档中找到足够信息，无法回答"
    assert response.sources == []
    retrieve_chunk.assert_awaited_once_with(
        user_query="没有答案的问题",
        user_id=3,
        document_id=None,
        db=db,
        top_k=7,
        score_threshold=0.66,
    )
    rerank_chunks.assert_not_awaited()
    response_user_query.assert_not_awaited()


async def test_query_returns_model_answer(monkeypatch):
    retrieve_results = [make_retrieve_item()]
    reranked_results = [
        retrieve_results[0].model_copy(update={"score": 0.97})
    ]
    retrieve_chunk = AsyncMock(return_value=retrieve_results)
    rerank_chunks = AsyncMock(return_value=reranked_results)
    response_user_query = AsyncMock(return_value="这是模型回答")

    monkeypatch.setattr(router, "retrieve_chunk", retrieve_chunk)
    monkeypatch.setattr(router, "rerank_chunks", rerank_chunks)
    monkeypatch.setattr(router, "response_user_query", response_user_query)

    response = await router.query(
        request=QueryRequest(
            user_query="测试问题",
            document_id=9,
            top_k=8,
            top_n=3,
            score_threshold=0.75,
        ),
        user_info={"user_id": 5},
        db=object(),
    )

    assert isinstance(response, QueryResponse)
    assert response.answer == "这是模型回答"
    assert len(response.sources) == 1
    assert response.sources[0].filename == "test.pdf"
    assert response.sources[0].page_number == 2
    assert response.sources[0].score == 0.97
    rerank_chunks.assert_awaited_once_with(
        "测试问题",
        retrieve_results,
        3,
    )
    response_user_query.assert_awaited_once_with(
        "测试问题",
        reranked_results,
    )


async def test_query_deduplicates_candidates_before_rerank(monkeypatch):
    retrieve_results = [
        make_item("重复文本", 0.70, 1, 1),
        make_item("重复文本", 0.95, 1, 1),
        make_item("重复文本", 0.90, 2, 3),
        make_item("唯一文本", 0.80, 1, 4),
    ]
    expected_candidates = [
        retrieve_results[1],
        retrieve_results[3],
    ]
    retrieve_chunk = AsyncMock(return_value=retrieve_results)
    rerank_chunks = AsyncMock(return_value=expected_candidates)
    response_user_query = AsyncMock(return_value="去重后的回答")

    monkeypatch.setattr(router, "retrieve_chunk", retrieve_chunk)
    monkeypatch.setattr(router, "rerank_chunks", rerank_chunks)
    monkeypatch.setattr(router, "response_user_query", response_user_query)

    response = await router.query(
        request=QueryRequest(
            user_query="去重测试",
            top_k=10,
            top_n=2,
        ),
        user_info={"user_id": 1},
        db=object(),
    )

    rerank_chunks.assert_awaited_once_with(
        "去重测试",
        expected_candidates,
        2,
    )
    response_user_query.assert_awaited_once_with(
        "去重测试",
        expected_candidates,
    )
    assert response.answer == "去重后的回答"


def test_query_request_validation():
    with pytest.raises(ValueError):
        QueryRequest(user_query="", top_k=5, score_threshold=0.5)

    with pytest.raises(ValueError):
        QueryRequest(user_query="问题", top_k=0, score_threshold=0.5)

    with pytest.raises(ValueError):
        QueryRequest(user_query="问题", top_k=5, score_threshold=1.1)

    with pytest.raises(ValueError, match="top_n 不能大于 top_k"):
        QueryRequest(
            user_query="问题",
            top_k=3,
            top_n=4,
            score_threshold=0.5,
        )
