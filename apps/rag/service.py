import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from apps.rag.crud import create_document_chunk
from apps.rag.embedding import embedding_documents
from apps.rag.parser import parse_document
from apps.rag.schemas import QuerySource, QueryResponse, RetrieveItem
from apps.rag.splitter import split_pages
from apps.rag.storage import download_to_temp
from apps.rag.vector_store import upsert_chunks
from models.documents import Document



async def ingest_document(user_id: int,document: Document,
                          db: AsyncSession):

    document.status = 'processing'
    temp_path = await download_to_temp(
        document.file_path,
        Path(document.filename).suffix,
    )

    try:
        pages = await parse_document(user_id,path=str(temp_path),document=document)
        chunks = await split_pages(pages)
        vectors = await embedding_documents(chunks)
        await upsert_chunks(vectors)
        chunks_objects = await create_document_chunk(chunks,db)
        document.chunk_count = len(chunks_objects)
        document.status = 'completed'
        document.error_message = None

    finally:
        await asyncio.to_thread(temp_path.unlink, missing_ok=True)

def organize_response(
    conversation_id: str,
    answer: str,
    rerank_results: list[RetrieveItem],
) -> QueryResponse:
    sources = [QuerySource(
            filename=item.metadata.filename,
            page_number=item.metadata.page_number,
            score=item.score,)
            for item in rerank_results
    ]
    return QueryResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=sources,
    )
