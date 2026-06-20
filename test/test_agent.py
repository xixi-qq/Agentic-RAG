from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from apps.rag import agent
from apps.rag.schemas import RetrieveItem, RetrieveItemMetadata


def make_retrieve_item(
    content: str = "RAG 使用检索结果辅助模型回答。",
    score: float = 0.91,
    filename: str = "rag-guide.pdf",
    page_number: int | None = 3,
    chunk_index: int = 1,
) -> RetrieveItem:
    return RetrieveItem(
        content=content,
        score=score,
        metadata=RetrieveItemMetadata(
            document_id=1,
            page_number=page_number,
            filename=filename,
            chunk_index=chunk_index,
        ),
    )


def test_get_content_formats_sources_and_text():
    content = agent.get_content(
        [
            make_retrieve_item(),
            make_retrieve_item(
                content="没有页码的文本。",
                score=0.72,
                filename="notes.txt",
                page_number=None,
            ),
        ]
    )

    assert "filename: rag-guide.pdf; page: 3" in content
    assert "0.91" in content
    assert "RAG 使用检索结果辅助模型回答。" in content
    assert "filename: notes.txt; page: None" in content
    assert "\n\n" in content


async def test_response_user_query_calls_async_model(monkeypatch):
    ainvoke = AsyncMock(
        return_value=SimpleNamespace(content="根据文档，RAG 会先检索再生成。")
    )
    monkeypatch.setattr(
        agent,
        "model",
        SimpleNamespace(ainvoke=ainvoke),
    )

    answer = await agent.response_user_query(
        "RAG 是怎么工作的？",
        [make_retrieve_item()],
    )

    assert answer == "根据文档，RAG 会先检索再生成。"
    ainvoke.assert_awaited_once()
    prompt = ainvoke.await_args.args[0]
    assert "RAG 是怎么工作的？" in prompt
    assert "rag-guide.pdf" in prompt
    assert "RAG 使用检索结果辅助模型回答。" in prompt


async def test_response_user_query_propagates_model_error(monkeypatch):
    ainvoke = AsyncMock(side_effect=RuntimeError("模型不可用"))
    monkeypatch.setattr(
        agent,
        "model",
        SimpleNamespace(ainvoke=ainvoke),
    )

    with pytest.raises(RuntimeError, match="模型不可用"):
        await agent.response_user_query(
            "测试问题",
            [make_retrieve_item()],
        )
