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

def test_get_content_returns_empty_string_without_results():
    assert agent.get_content([]) == ""
