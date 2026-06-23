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
        score=0.97,
        metadata=RetrieveItemMetadata(
            document_id=9,
            page_number=2,
            filename="test.pdf",
            chunk_index=4,
        ),
    )


async def test_query_returns_fallback_without_calling_model(monkeypatch):
    retrieve_and_rerank = AsyncMock(return_value=[])
    response_user_query = AsyncMock()
    db = object()

    monkeypatch.setattr(
        router,
        "retrieve_and_rerank",
        retrieve_and_rerank,
    )
    monkeypatch.setattr(
        router,
        "response_user_query",
        response_user_query,
    )

    response = await router.query(
        request=QueryRequest(
            user_query="没有答案的问题",
            document_id=None,
            candidate_k=7,
            final_k=3,
            score_threshold=0.66,
        ),
        user_info={"user_id": 3},
        db=db,
    )

    assert isinstance(response, QueryResponse)
    assert response.answer == "未在文档中找到足够信息，无法回答"
    assert response.sources == []
    retrieve_and_rerank.assert_awaited_once_with(
        query="没有答案的问题",
        user_id=3,
        document_id=None,
        db=db,
        candidate_k=7,
        final_k=3,
        score_threshold=0.66,
    )
    response_user_query.assert_not_awaited()


async def test_query_returns_model_answer(monkeypatch):
    reranked_results = [make_retrieve_item()]
    retrieve_and_rerank = AsyncMock(
        return_value=reranked_results,
    )
    response_user_query = AsyncMock(
        return_value="这是模型回答",
    )
    db = object()

    monkeypatch.setattr(
        router,
        "retrieve_and_rerank",
        retrieve_and_rerank,
    )
    monkeypatch.setattr(
        router,
        "response_user_query",
        response_user_query,
    )

    response = await router.query(
        request=QueryRequest(
            user_query="测试问题",
            document_id=9,
            candidate_k=8,
            final_k=3,
            score_threshold=0.75,
        ),
        user_info={"user_id": 5},
        db=db,
    )

    assert isinstance(response, QueryResponse)
    assert response.answer == "这是模型回答"
    assert len(response.sources) == 1
    assert response.sources[0].filename == "test.pdf"
    assert response.sources[0].page_number == 2
    assert response.sources[0].score == 0.97
    retrieve_and_rerank.assert_awaited_once_with(
        query="测试问题",
        user_id=5,
        document_id=9,
        db=db,
        candidate_k=8,
        final_k=3,
        score_threshold=0.75,
    )
    response_user_query.assert_awaited_once_with(
        "测试问题",
        reranked_results,
    )


def test_query_request_validation():
    with pytest.raises(ValueError):
        QueryRequest(
            user_query="",
            candidate_k=5,
            score_threshold=0.5,
        )

    with pytest.raises(ValueError):
        QueryRequest(
            user_query="问题",
            candidate_k=0,
            score_threshold=0.5,
        )

    with pytest.raises(ValueError):
        QueryRequest(
            user_query="问题",
            candidate_k=5,
            score_threshold=1.1,
        )

    with pytest.raises(
        ValueError,
        match="final_k 不能大于 candidate_k",
    ):
        QueryRequest(
            user_query="问题",
            candidate_k=3,
            final_k=4,
            score_threshold=0.5,
        )
