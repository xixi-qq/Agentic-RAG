from core.queue import rag_queue
from apps.rag.tasks import process_document_task


def enqueue_document_ingestion(user_id: int, document_id: int):
    return rag_queue.enqueue(
        process_document_task,
        user_id,
        document_id,
    )