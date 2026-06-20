from fastapi import UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.rag.schemas import Chunk
from config.minio_config import MINIO_BUCKET
from models.documents import Document, DocumentChunk


async def create_document(user_info,file: UploadFile,db: AsyncSession,object_name,file_size,file_hash) -> Document:

    document = Document(user_id=user_info['user_id'],
                        filename=file.filename,
                        content_type=file.content_type,
                        status='pending',
                        file_hash=file_hash,
                        file_path=object_name,
                        bucket_name=MINIO_BUCKET,
                        file_size=file_size,
                        error_message= None,
                        chunk_count=0)
    db.add(document)
    await db.flush()
    await db.refresh(document)
    return document



async def get_all_documents_by_user(user_id: int,db: AsyncSession):
    stmt = select(Document).where(Document.user_id == user_id).order_by(Document.created_at)
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_document_by_id(document_id: int,db: AsyncSession):
    stmt = select(Document).where(Document.id == document_id)
    result = await db.execute(stmt)
    return result.scalars().first()


async def get_document_by_hash(user_id,file_hash,db: AsyncSession):
    stmt = select(Document).where(Document.file_hash == file_hash,Document.user_id == user_id)
    result = await db.execute(stmt)
    return result.scalars().first()


async def create_document_chunk(chunks: list[Chunk],
                                db: AsyncSession) -> list[DocumentChunk]:

    chunks = [
        DocumentChunk(document_id=chunk.document_id,
                      chunk_index=chunk.chunk_index,
                      content=chunk.content,
                      vector_id=chunk.vector_id,
                      page_number=chunk.page_number)
                      for chunk in chunks
    ]
    db.add_all(chunks)
    await db.flush()
    return chunks


async def delete_document_chunks(document_id: int, db: AsyncSession) -> None:
    stmt = delete(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
    )
    await db.execute(stmt)



async def get_chunk_by_vector_id(user_id, vectors, db: AsyncSession):
    query = (
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.vector_id.in_(vectors),
            Document.user_id == user_id,
        )
    )
    res = await db.execute(query)
    return res.all()
