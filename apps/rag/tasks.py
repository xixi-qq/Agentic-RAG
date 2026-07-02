import asyncio
import logging
from apps.rag.bm25 import bm25_cache
from apps.rag.crud import delete_document_chunks, get_document_by_id
from apps.rag.service import ingest_document
from apps.rag.vector_store import delete_vectors
from config.db_config import AsyncSessionLocal


logger = logging.getLogger(__name__)


def process_document_task(user_id: int, document_id: int) -> None:
    asyncio.run(process_document(user_id, document_id))


async def process_document(user_id: int, document_id: int) -> None:
    async with AsyncSessionLocal() as db:
        document = await get_document_by_id(document_id, db)
        if not document:
            logger.warning("document ingestion skipped: document_id=%s not found", document_id)
            return

        if document.user_id != user_id:
            logger.warning(
                "document ingestion skipped: document_id=%s belongs to user_id=%s, not %s",
                document_id,
                document.user_id,
                user_id,
            )
            return

        try:
            await delete_document_chunks(document_id, db)
            await delete_vectors(document_id)
            document.status = "processing"
            document.error_message = None
            document.chunk_count = 0
            await db.commit()

            await ingest_document(user_id, document, db)
            await db.commit()

            await bm25_cache.invalidate(
                user_id=user_id,
                document_id=document_id,
            )
            logger.info("document ingestion completed: document_id=%s", document_id)

        except Exception as exc:
            await db.rollback()
            logger.exception("document ingestion failed: document_id=%s", document_id)

            try:
                await delete_vectors(document_id)
                await delete_document_chunks(document_id, db)

                failed_document = await get_document_by_id(document_id, db)
                if failed_document:
                    failed_document.status = "failed"
                    failed_document.error_message = f"{type(exc).__name__}: {str(exc)[:1000]}"
                    failed_document.chunk_count = 0
                await db.commit()
                await bm25_cache.invalidate(
                    user_id=user_id,
                    document_id=document_id,
                )
            except Exception:
                await db.rollback()
                logger.exception(
                    "document ingestion failure cleanup failed: document_id=%s",
                    document_id,
                )
            raise
