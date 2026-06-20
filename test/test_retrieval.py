from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client.models import MatchValue

from apps.rag import retrieval
from apps.rag.schemas import RetrieveItem, RetrieveItemMetadata


def make_point(vector_id: str, score: float):
    return SimpleNamespace(id=vector_id, score=score)


def make_chunk(
    vector_id: str,
    content: str,
    document_id: int,
    page_number: int | None,
    filename: str = "test-document.txt",
    chunk_index: int = 1,
):
    return SimpleNamespace(
        vector_id=vector_id,
        content=content,
        document_id=document_id,
        page_number=page_number,
        filename=filename,
        chunk_index=chunk_index,
    )


def make_retrieve_item(
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
            chunk_index=chunk_index,
            filename="test.txt",
        ),
    )


def test_deduplicate_chunks_keeps_highest_score():
    items = [
        make_retrieve_item("旧结果", 0.61, 1, 2),
        make_retrieve_item("高分结果", 0.93, 1, 2),
        make_retrieve_item("其他结果", 0.82, 1, 3),
    ]

    result = retrieval.deduplicate_chunks(items)

    assert [item.content for item in result] == ["高分结果", "其他结果"]
    assert [item.score for item in result] == [0.93, 0.82]


def test_deduplicate_chunks_keeps_same_index_from_different_documents():
    items = [
        make_retrieve_item("文档一", 0.8, 1, 2),
        make_retrieve_item("文档二", 0.7, 2, 2),
    ]

    result = retrieval.deduplicate_chunks(items)

    assert len(result) == 2


def test_deduplicate_by_content_normalizes_whitespace_and_keeps_highest_score():
    items = [
        make_retrieve_item("RAG   检索\n增强", 0.71, 1, 1),
        make_retrieve_item("RAG 检索 增强", 0.92, 2, 3),
        make_retrieve_item("另一段文本", 0.83, 1, 4),
    ]

    result = retrieval.deduplicate_by_content(items)

    assert [item.content for item in result] == [
        "RAG 检索 增强",
        "另一段文本",
    ]
    assert [item.score for item in result] == [0.92, 0.83]


async def test_search_vector_filters_by_user(monkeypatch):
    embed_query = AsyncMock(return_value=[0.1, 0.2])
    query_points = AsyncMock(
        return_value=SimpleNamespace(points=[make_point("vector-1", 0.91)])
    )

    monkeypatch.setattr(retrieval, "embed_query", embed_query)
    monkeypatch.setattr(retrieval.client, "query_points", query_points)

    points = await retrieval.search_vector(
        query="RAG 是什么",
        user_id=7,
        document_id=None,
        score_threshold=0.5,
        top_k=3,
    )

    assert len(points) == 1
    embed_query.assert_awaited_once_with("RAG 是什么")

    kwargs = query_points.await_args.kwargs
    assert kwargs["collection_name"] == retrieval.collection_name
    assert kwargs["query"] == [0.1, 0.2]
    assert kwargs["limit"] == 3
    assert kwargs["score_threshold"] == 0.5
    assert kwargs["with_payload"] is True
    assert kwargs["with_vectors"] is False

    conditions = kwargs["query_filter"].must
    assert len(conditions) == 1
    assert conditions[0].key == "user_id"
    assert conditions[0].match == MatchValue(value=7)


async def test_search_vector_filters_by_document(monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "embed_query",
        AsyncMock(return_value=[0.1, 0.2]),
    )
    query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    monkeypatch.setattr(retrieval.client, "query_points", query_points)

    await retrieval.search_vector(
        query="指定文档问题",
        user_id=7,
        document_id=12,
        score_threshold=0.6,
        top_k=5,
    )

    conditions = query_points.await_args.kwargs["query_filter"].must
    assert len(conditions) == 2
    assert conditions[0].key == "user_id"
    assert conditions[0].match == MatchValue(value=7)
    assert conditions[1].key == "document_id"
    assert conditions[1].match == MatchValue(value=12)


async def test_retrieve_chunk_returns_empty_when_no_vector_matches(monkeypatch):
    search_vector = AsyncMock(return_value=[])
    get_chunks = AsyncMock()

    monkeypatch.setattr(retrieval, "search_vector", search_vector)
    monkeypatch.setattr(retrieval, "get_chunk_by_vector_id", get_chunks)

    result = await retrieval.retrieve_chunk(
        user_query="没有答案的问题",
        user_id=3,
        document_id=None,
        db=object(),
    )

    assert result == []
    search_vector.assert_awaited_once_with(
        "没有答案的问题",
        3,
        None,
        0.5,
        15,
    )
    get_chunks.assert_not_awaited()


async def test_retrieve_chunk_preserves_vector_score_order(monkeypatch):
    points = [
        make_point("vector-high", 0.95),
        make_point("vector-low", 0.72),
    ]
    chunk_rows = [
        (
            make_chunk(
                "vector-low",
                "低分内容",
                2,
                None,
                chunk_index=8,
            ),
            "low.txt",
        ),
        (
            make_chunk(
                "vector-high",
                "高分内容",
                1,
                4,
                chunk_index=3,
            ),
            "high.txt",
        ),
    ]
    db = object()

    monkeypatch.setattr(
        retrieval,
        "search_vector",
        AsyncMock(return_value=points),
    )
    get_chunks = AsyncMock(return_value=chunk_rows)
    monkeypatch.setattr(retrieval, "get_chunk_by_vector_id", get_chunks)

    result = await retrieval.retrieve_chunk(
        user_query="排序问题",
        user_id=3,
        document_id=None,
        db=db,
    )

    get_chunks.assert_awaited_once_with(
        3,
        ["vector-high", "vector-low"],
        db,
    )
    assert [item.content for item in result] == ["高分内容", "低分内容"]
    assert [item.score for item in result] == [0.95, 0.72]
    assert result[0].metadata.document_id == 1
    assert result[0].metadata.page_number == 4
    assert result[0].metadata.filename == "high.txt"
    assert result[0].metadata.chunk_index == 3
    assert result[1].metadata.document_id == 2
    assert result[1].metadata.page_number is None
    assert result[1].metadata.filename == "low.txt"
    assert result[1].metadata.chunk_index == 8


async def test_retrieve_chunk_forwards_search_options(monkeypatch):
    search_vector = AsyncMock(return_value=[])
    monkeypatch.setattr(retrieval, "search_vector", search_vector)

    await retrieval.retrieve_chunk(
        user_query="自定义检索参数",
        user_id=8,
        document_id=13,
        db=object(),
        top_k=9,
        score_threshold=0.72,
    )

    search_vector.assert_awaited_once_with(
        "自定义检索参数",
        8,
        13,
        0.72,
        9,
    )


async def test_retrieve_chunk_skips_missing_database_chunk(monkeypatch):
    points = [
        make_point("existing-vector", 0.9),
        make_point("missing-vector", 0.8),
    ]
    chunk_rows = [
        (make_chunk("existing-vector", "存在的内容", 1, 2), "existing.txt"),
    ]

    monkeypatch.setattr(
        retrieval,
        "search_vector",
        AsyncMock(return_value=points),
    )
    monkeypatch.setattr(
        retrieval,
        "get_chunk_by_vector_id",
        AsyncMock(return_value=chunk_rows),
    )

    result = await retrieval.retrieve_chunk(
        user_query="数据不一致",
        user_id=3,
        document_id=1,
        db=object(),
    )

    assert len(result) == 1
    assert result[0].content == "存在的内容"
    assert result[0].metadata.filename == "existing.txt"


@pytest.mark.parametrize("document_id", [0, None])
async def test_search_vector_does_not_add_empty_document_filter(
    monkeypatch,
    document_id,
):
    monkeypatch.setattr(
        retrieval,
        "embed_query",
        AsyncMock(return_value=[0.1, 0.2]),
    )
    query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    monkeypatch.setattr(retrieval.client, "query_points", query_points)

    await retrieval.search_vector(
        query="测试",
        user_id=1,
        document_id=document_id,
        score_threshold=0.5,
    )

    conditions = query_points.await_args.kwargs["query_filter"].must
    assert len(conditions) == 1
