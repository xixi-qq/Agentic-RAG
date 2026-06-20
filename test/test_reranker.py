from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.rag import reranker
from apps.rag.schemas import RetrieveItem, RetrieveItemMetadata


def make_item(content: str, score: float, chunk_index: int) -> RetrieveItem:
    return RetrieveItem(
        content=content,
        score=score,
        metadata=RetrieveItemMetadata(
            document_id=1,
            page_number=1,
            chunk_index=chunk_index,
            filename="test.pdf",
        ),
    )


class FakeAsyncClient:
    def __init__(self, response):
        self.response = response
        self.post = AsyncMock(return_value=response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


async def test_rerank_empty_items_does_not_call_api(monkeypatch):
    async_client = AsyncMock()
    monkeypatch.setattr(reranker.httpx, "AsyncClient", async_client)

    result = await reranker.rerank_chunks("问题", [], top_n=5)

    assert result == []
    async_client.assert_not_called()


async def test_rerank_reorders_items_and_updates_scores(monkeypatch):
    items = [
        make_item("第一段", 0.8, 1),
        make_item("第二段", 0.7, 2),
        make_item("第三段", 0.6, 3),
    ]
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "results": [
                {"index": 2, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.83},
            ]
        },
    )
    fake_client = FakeAsyncClient(response)
    monkeypatch.setattr(
        reranker.httpx,
        "AsyncClient",
        lambda timeout: fake_client,
    )

    result = await reranker.rerank_chunks(
        query="测试问题",
        items=items,
        top_n=2,
    )

    assert [item.content for item in result] == ["第三段", "第一段"]
    assert [item.score for item in result] == [0.98, 0.83]
    assert items[2].score == 0.6

    request = fake_client.post.await_args
    assert request.args[0] == reranker.RERANK_URL
    assert request.kwargs["json"]["query"] == "测试问题"
    assert request.kwargs["json"]["documents"] == [
        "第一段",
        "第二段",
        "第三段",
    ]
    assert request.kwargs["json"]["top_n"] == 2


async def test_rerank_caps_top_n_to_item_count(monkeypatch):
    items = [make_item("唯一候选", 0.5, 1)]
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "results": [
                {"index": 0, "relevance_score": 0.9},
            ]
        },
    )
    fake_client = FakeAsyncClient(response)
    monkeypatch.setattr(
        reranker.httpx,
        "AsyncClient",
        lambda timeout: fake_client,
    )

    await reranker.rerank_chunks("问题", items, top_n=5)

    assert fake_client.post.await_args.kwargs["json"]["top_n"] == 1


async def test_rerank_propagates_http_error(monkeypatch):
    def raise_error():
        raise RuntimeError("rerank 服务不可用")

    response = SimpleNamespace(
        raise_for_status=raise_error,
        json=lambda: {},
    )
    fake_client = FakeAsyncClient(response)
    monkeypatch.setattr(
        reranker.httpx,
        "AsyncClient",
        lambda timeout: fake_client,
    )

    with pytest.raises(RuntimeError, match="rerank 服务不可用"):
        await reranker.rerank_chunks(
            "问题",
            [make_item("候选", 0.5, 1)],
        )
