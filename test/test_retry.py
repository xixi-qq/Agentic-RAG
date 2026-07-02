from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
        error_message="old error",
        chunk_count=3,
    )


async def test_retry_failed_document_enqueues_ingestion(monkeypatch):
    db = FakeDB()
    document = make_document()
    enqueue_mock = Mock()

    monkeypatch.setattr(
        router,
        "get_document_by_id",
        AsyncMock(return_value=document),
    )
    monkeypatch.setattr(router, "enqueue_document_ingestion", enqueue_mock)

    result = await router.retry_document(
        document_id=9,
        user_info={"user_id": 1},
        db=db,
    )

    assert result["status"] == "pending"
    assert result["chunk_count"] == 0
    assert result["error_message"] is None
    enqueue_mock.assert_called_once_with(1, 9)
    assert db.commits == 1
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


async def test_retry_enqueue_failure_restores_failed_status(monkeypatch):
    db = FakeDB()
    document = make_document()

    monkeypatch.setattr(
        router,
        "get_document_by_id",
        AsyncMock(return_value=document),
    )
    monkeypatch.setattr(
        router,
        "enqueue_document_ingestion",
        Mock(side_effect=RuntimeError("Redis unavailable")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await router.retry_document(
            document_id=9,
            user_info={"user_id": 1},
            db=db,
        )

    assert exc_info.value.status_code == 422
    assert document.status == "failed"
    assert "Redis unavailable" in document.error_message
    assert document.chunk_count == 0
    assert db.commits == 1
    assert db.rollbacks == 1
