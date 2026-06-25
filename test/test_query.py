from types import SimpleNamespace
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
    ainvoke = AsyncMock(
        return_value={
            "answer": "未在文档中找到足够信息，无法回答",
            "candidates": [],
        }
    )
    db = object()

    create_conversation = AsyncMock(
        return_value=SimpleNamespace(id="conversation-1", title=None),
    )
    add_message = AsyncMock()
    generate_title = AsyncMock(return_value="没有答案的问题")
    update_title = AsyncMock()
    monkeypatch.setattr(router, "create_conversation", create_conversation)
    monkeypatch.setattr(router, "add_message", add_message)
    monkeypatch.setattr(router, "generate_conversation_title", generate_title)
    monkeypatch.setattr(router, "update_conversation_title", update_title)

    response = await router.query(
        request_app=SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    rag_graph=SimpleNamespace(ainvoke=ainvoke),
                ),
            ),
        ),
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
    assert response.conversation_id == "conversation-1"
    assert response.answer == "未在文档中找到足够信息，无法回答"
    assert response.sources == []

    call = ainvoke.await_args
    assert call.args[0] == {
        "messages": [
            {
                "role": "user",
                "content": "没有答案的问题",
            }
        ],
        "original_query": "没有答案的问题",
        "search_query": "没有答案的问题",
        "rewrite_count": 0,
    }
    assert call.kwargs["config"] == {
        "configurable": {
            "thread_id": "conversation-1",
        },
    }
    context = call.kwargs["context"]
    assert context.user_id == 3
    assert context.document_id is None
    assert context.db is db
    assert context.retrieval.candidate_k == 7
    assert context.retrieval.final_k == 3
    assert context.retrieval.score_threshold == 0.66
    assert add_message.await_count == 2
    generate_title.assert_awaited_once_with(
        "没有答案的问题",
        "未在文档中找到足够信息，无法回答",
    )
    update_title.assert_awaited_once_with(
        db,
        3,
        "conversation-1",
        "没有答案的问题",
    )


async def test_query_returns_model_answer(monkeypatch):
    reranked_results = [make_retrieve_item()]
    ainvoke = AsyncMock(
        return_value={
            "answer": "这是模型回答",
            "candidates": reranked_results,
        }
    )
    db = object()

    create_conversation = AsyncMock(
        return_value=SimpleNamespace(id="conversation-2", title=None),
    )
    add_message = AsyncMock()
    generate_title = AsyncMock(return_value="测试问题")
    update_title = AsyncMock()
    monkeypatch.setattr(router, "create_conversation", create_conversation)
    monkeypatch.setattr(router, "add_message", add_message)
    monkeypatch.setattr(router, "generate_conversation_title", generate_title)
    monkeypatch.setattr(router, "update_conversation_title", update_title)

    response = await router.query(
        request_app=SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    rag_graph=SimpleNamespace(ainvoke=ainvoke),
                ),
            ),
        ),
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
    assert response.conversation_id == "conversation-2"
    assert response.answer == "这是模型回答"
    assert len(response.sources) == 1
    assert response.sources[0].filename == "test.pdf"
    assert response.sources[0].page_number == 2
    assert response.sources[0].score == 0.97

    call = ainvoke.await_args
    assert call.args[0] == {
        "messages": [
            {
                "role": "user",
                "content": "测试问题",
            }
        ],
        "original_query": "测试问题",
        "search_query": "测试问题",
        "rewrite_count": 0,
    }
    assert call.kwargs["config"] == {
        "configurable": {
            "thread_id": "conversation-2",
        },
    }
    context = call.kwargs["context"]
    assert context.user_id == 5
    assert context.document_id == 9
    assert context.db is db
    assert context.retrieval.candidate_k == 8
    assert context.retrieval.final_k == 3
    assert context.retrieval.score_threshold == 0.75
    assert add_message.await_count == 2
    generate_title.assert_awaited_once_with(
        "测试问题",
        "这是模型回答",
    )
    update_title.assert_awaited_once_with(
        db,
        5,
        "conversation-2",
        "测试问题",
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
