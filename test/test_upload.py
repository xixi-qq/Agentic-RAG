import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

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


def make_upload_file(content=b"hello RAG", content_type="text/plain"):
    return UploadFile(
        filename="example.txt",
        file=BytesIO(content),
        headers={"content-type": content_type},
    )


def make_document(document_id=1, status="pending"):
    return SimpleNamespace(
        id=document_id,
        user_id=1,
        filename="example.txt",
        content_type="text/plain",
        status=status,
        file_hash="abc",
        file_path="users/1/example.txt",
        error_message=None,
        chunk_count=0,
    )


async def test_upload_success(monkeypatch):
    db = FakeDB()
    document = make_document()

    monkeypatch.setattr(router, "calculate_file_hash", AsyncMock(return_value="abc"))
    monkeypatch.setattr(router, "get_document_by_hash", AsyncMock(return_value=None))
    monkeypatch.setattr(
        router,
        "upload_file",
        AsyncMock(return_value=("users/1/example.txt", 9)),
    )
    monkeypatch.setattr(router, "create_document", AsyncMock(return_value=document))

    async def complete_ingestion(_user_id, doc, _db):
        doc.status = "completed"
        doc.chunk_count = 1

    monkeypatch.setattr(router, "ingest_document", complete_ingestion)

    response = await router.upload_document(
        user_info={"user_id": 1},
        file=make_upload_file(),
        db=db,
    )
    body = json.loads(response.body)

    assert response.status_code == 201
    assert body["status"] == "completed"
    assert body["chunk_count"] == 1
    assert db.commits == 2
    assert db.rollbacks == 0


async def test_duplicate_completed_document_is_rejected(monkeypatch):
    db = FakeDB()
    existing = make_document(status="completed")

    monkeypatch.setattr(router, "calculate_file_hash", AsyncMock(return_value="abc"))
    monkeypatch.setattr(
        router,
        "get_document_by_hash",
        AsyncMock(return_value=existing),
    )

    with pytest.raises(HTTPException) as exc_info:
        await router.upload_document(
            user_info={"user_id": 1},
            file=make_upload_file(),
            db=db,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "文件已存在"


async def test_duplicate_failed_document_returns_retry_information(monkeypatch):
    db = FakeDB()
    existing = make_document(document_id=7, status="failed")

    monkeypatch.setattr(router, "calculate_file_hash", AsyncMock(return_value="abc"))
    monkeypatch.setattr(
        router,
        "get_document_by_hash",
        AsyncMock(return_value=existing),
    )

    with pytest.raises(HTTPException) as exc_info:
        await router.upload_document(
            user_info={"user_id": 1},
            file=make_upload_file(),
            db=db,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["document_id"] == 7
    assert exc_info.value.detail["retryable"] is True


async def test_parse_failure_keeps_minio_file(monkeypatch):
    db = FakeDB()
    document = make_document(document_id=7)
    failed_document = make_document(document_id=7)
    delete_mock = AsyncMock()

    monkeypatch.setattr(router, "calculate_file_hash", AsyncMock(return_value="abc"))
    monkeypatch.setattr(router, "get_document_by_hash", AsyncMock(return_value=None))
    monkeypatch.setattr(
        router,
        "upload_file",
        AsyncMock(return_value=("users/1/example.txt", 3)),
    )
    monkeypatch.setattr(router, "create_document", AsyncMock(return_value=document))
    monkeypatch.setattr(
        router,
        "ingest_document",
        AsyncMock(side_effect=ValueError("解析失败")),
    )
    monkeypatch.setattr(
        router,
        "get_document_by_id",
        AsyncMock(return_value=failed_document),
    )
    monkeypatch.setattr(router, "delete_file", delete_mock)

    with pytest.raises(HTTPException) as exc_info:
        await router.upload_document(
            user_info={"user_id": 1},
            file=make_upload_file(b"bad"),
            db=db,
        )

    assert exc_info.value.status_code == 422
    assert failed_document.status == "failed"
    assert failed_document.error_message == "解析失败"
    assert failed_document.chunk_count == 0
    delete_mock.assert_not_awaited()
    assert db.commits == 2
    assert db.rollbacks == 1


async def test_metadata_failure_removes_uploaded_file(monkeypatch):
    db = FakeDB()
    delete_mock = AsyncMock()

    monkeypatch.setattr(router, "calculate_file_hash", AsyncMock(return_value="abc"))
    monkeypatch.setattr(router, "get_document_by_hash", AsyncMock(return_value=None))
    monkeypatch.setattr(
        router,
        "upload_file",
        AsyncMock(return_value=("users/1/orphan.txt", 3)),
    )
    monkeypatch.setattr(
        router,
        "create_document",
        AsyncMock(side_effect=RuntimeError("数据库失败")),
    )
    monkeypatch.setattr(router, "delete_file", delete_mock)

    with pytest.raises(HTTPException) as exc_info:
        await router.upload_document(
            user_info={"user_id": 1},
            file=make_upload_file(),
            db=db,
        )

    assert exc_info.value.status_code == 500
    delete_mock.assert_awaited_once_with("users/1/orphan.txt")
    assert db.rollbacks == 1


async def test_file_too_large_keeps_413(monkeypatch):
    db = FakeDB()

    monkeypatch.setattr(router, "calculate_file_hash", AsyncMock(return_value="abc"))
    monkeypatch.setattr(router, "get_document_by_hash", AsyncMock(return_value=None))
    monkeypatch.setattr(
        router,
        "upload_file",
        AsyncMock(
            side_effect=HTTPException(
                status_code=413,
                detail="文件大小不能超过 20MB",
            )
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await router.upload_document(
            user_info={"user_id": 1},
            file=make_upload_file(),
            db=db,
        )

    assert exc_info.value.status_code == 413
    assert db.rollbacks == 1
