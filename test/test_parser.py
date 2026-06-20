from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.rag.parser import parse_document


def temporary_test_file(suffix: str) -> Path:
    return Path(__file__).parent / f".tmp_{uuid4().hex}{suffix}"


def make_document(content_type: str):
    return SimpleNamespace(
        id=1,
        content_type=content_type,
        filename="test-document.txt",
    )


async def test_parse_text():
    path = temporary_test_file(".txt")
    try:
        path.write_text("hello RAG", encoding="utf-8")

        pages = await parse_document(
            user_id=1,
            path=str(path),
            document=make_document("text/plain"),
        )

        assert len(pages) == 1
        assert pages[0].user_id == 1
        assert pages[0].document_id == 1
        assert pages[0].page_number == 1
        assert pages[0].content == "hello RAG"
        assert pages[0].filename == "test-document.txt"
    finally:
        path.unlink(missing_ok=True)


async def test_parse_markdown():
    path = temporary_test_file(".md")
    try:
        path.write_text("# 标题\n\n正文", encoding="utf-8")

        pages = await parse_document(
            user_id=1,
            path=str(path),
            document=make_document("text/markdown"),
        )

        assert len(pages) == 1
        assert pages[0].content == "# 标题\n\n正文"
    finally:
        path.unlink(missing_ok=True)


async def test_parse_empty_text_raises_error():
    path = temporary_test_file(".txt")
    try:
        path.write_text("   \n", encoding="utf-8")

        with pytest.raises(ValueError, match="有效文本"):
            await parse_document(
                user_id=1,
                path=str(path),
                document=make_document("text/plain"),
            )
    finally:
        path.unlink(missing_ok=True)


async def test_parse_unsupported_type_raises_error():
    path = temporary_test_file(".png")
    try:
        path.write_bytes(b"not-an-image")

        with pytest.raises(ValueError, match="不支持的文件类型"):
            await parse_document(
                user_id=1,
                path=str(path),
                document=make_document("image/png"),
            )
    finally:
        path.unlink(missing_ok=True)
