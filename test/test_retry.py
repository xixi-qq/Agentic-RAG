from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from apps.rag import router


class FakeDB:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0
        self.refreshes = 0

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1

    async def refresh(self, _):
        self.refreshes += 1


def make_document(status="failed", user_id=1):
    return SimpleNamespace(
        id=9,
        user_id=user_id,
        status=status,
        error_message="旧错误",
        chunk_count=3,
    )


async def test_retry_failed_document_success(monkeypatch):
    db = FakeDB()
    document = make_document()
    cleanup_mock = AsyncMock()
    vector_cleanup_mock = AsyncMock()

    monkeypatch.setattr(
        router,
        "get_document_by_id",
        AsyncMock(return_value=document),
    )
    monkeypatch.setattr(router, "delete_document_chunks", cleanup_mock)
    monkeypatch.setattr(router, "delete_vectors", vector_cleanup_mock)

    async def complete_ingestion(_user_id, doc, _db):
        doc.status = "completed"
        doc.error_message = None
        doc.chunk_count = 2

    monkeypatch.setattr(router, "ingest_document", complete_ingestion)

    result = await router.retry_document(
        document_id=9,
        user_info={"user_id": 1},
        db=db,
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] == 2
    assert result["error_message"] is None
    cleanup_mock.assert_awaited_once_with(9, db)
    vector_cleanup_mock.assert_awaited_once_with(9)
    assert db.commits == 2
    assert db.rollbacks == 0


async def test_retry_missing_document_returns_404(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(
        router,
        "get_document_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(HTTPException) as exc_info:
        await router.retry_document(
            document_id=999,
            user_info={"user_id": 1},
            db=db,
        )

    assert exc_info.value.status_code == 404


async def test_retry_other_users_document_returns_403(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(
        router,
        "get_document_by_id",
        AsyncMock(return_value=make_document(user_id=2)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await router.retry_document(
            document_id=9,
            user_info={"user_id": 1},
            db=db,
        )

    assert exc_info.value.status_code == 403


async def test_retry_non_failed_document_returns_409(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(
        router,
        "get_document_by_id",
        AsyncMock(return_value=make_document(status="completed")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await router.retry_document(
            document_id=9,
            user_info={"user_id": 1},
            db=db,
        )

    assert exc_info.value.status_code == 409


async def test_retry_failure_restores_failed_status(monkeypatch):
    db = FakeDB()
    document = make_document()
    failed_document = make_document(status="processing")

    monkeypatch.setattr(
        router,
        "get_document_by_id",
        AsyncMock(side_effect=[document, failed_document]),
    )
    monkeypatch.setattr(router, "delete_document_chunks", AsyncMock())
    monkeypatch.setattr(router, "delete_vectors", AsyncMock())
    monkeypatch.setattr(
        router,
        "ingest_document",
        AsyncMock(side_effect=ValueError("再次解析失败")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await router.retry_document(
            document_id=9,
            user_info={"user_id": 1},
            db=db,
        )

    assert exc_info.value.status_code == 422
    assert failed_document.status == "failed"
    assert failed_document.error_message == "再次解析失败"
    assert failed_document.chunk_count == 0
    assert db.commits == 2
    assert db.rollbacks == 1
