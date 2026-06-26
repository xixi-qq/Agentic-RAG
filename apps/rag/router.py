import logging

from fastapi.responses import JSONResponse
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from apps.rag.bm25 import bm25_cache
from apps.rag.crud import (
    create_document,
    delete_document_chunks,
    get_all_documents_by_user,
    get_document_by_hash,
    get_document_by_id
)
from apps.rag.parser import calculate_file_hash
from apps.rag.schemas import QueryRequest, QueryResponse
from apps.rag.query_service import ask_rag
from apps.rag.service import ingest_document
from apps.rag.storage import upload_file, delete_file
from apps.rag.vector_store import delete_vectors
from config.db_config import get_db
from utils.jwt import get_current_user
from settings import ALLOWED_TYPES
from fastapi import Request

router = APIRouter(prefix='/rag',tags=['rag'])
logger = logging.getLogger(__name__)


async def cleanup_uploaded_file(object_name: str | None) -> None:
    if not object_name:
        return

    try:
        await delete_file(object_name)
    except Exception:
        logger.exception("清理 MinIO 对象失败: %s", object_name)


@router.post('/documents')
async def upload_document(
        user_info: dict = Depends(get_current_user),
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db)):
    """
    user_info : {
            "user_id": user.id,
            "username": user.name,
            "exp": expire
        }
        """

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400,detail='不支持的文件类型')
    file_hash = await calculate_file_hash(file)
    existing = await get_document_by_hash(user_info['user_id'],file_hash,db)
    if existing:
        if existing.status == "failed":
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "文件上次解析失败，可以执行重试",
                    "document_id": existing.id,
                    "retryable": True,
                },
            )
        raise HTTPException(status_code=409, detail="文件已存在")

    object_name = None
    metadata_committed = False
    try:
        object_name,filesize = await upload_file(user_info['user_id'],file)

        document = await create_document(
            user_info,
            file,
            db,
            object_name,
            filesize,
            file_hash,
        )
        await db.commit()
        metadata_committed = True
        await db.refresh(document)

        try:
            await ingest_document(user_info["user_id"],document,db)
            await db.commit()
            await db.refresh(document)
            await bm25_cache.invalidate(
                user_id=user_info["user_id"],
                document_id=document.id,
            )
        except Exception as exc:
            document_id = document.id
            await db.rollback()

            failed_document = await get_document_by_id(document_id, db)
            if failed_document:
                failed_document.status = "failed"
                failed_document.error_message = f"{type(exc).__name__}: {str(exc)[:1000]}"
                failed_document.chunk_count = 0
                await db.commit()

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "文档解析失败，可以稍后重试",
                    "document_id": document_id,
                    "retryable": True,
                },
            ) from exc

        response_data = {
            'id':document.id,
            'filename':document.filename,
            'content_type':document.content_type,
            'status':document.status,
            'file_hash':document.file_hash,
            'error_message':document.error_message,
            'chunk_count':document.chunk_count
        }
        return JSONResponse(status_code=201,content=response_data)

    except HTTPException:
        if not metadata_committed:
            await db.rollback()
            await cleanup_uploaded_file(object_name)
        raise
    except IntegrityError as exc:
        await db.rollback()
        await cleanup_uploaded_file(object_name)
        raise HTTPException(status_code=409, detail="文件已存在") from exc
    except Exception as exc:
        await db.rollback()
        if not metadata_committed:
            await cleanup_uploaded_file(object_name)
        raise HTTPException(status_code=500,detail="文件上传失败") from exc


@router.post('/documents/{document_id}/retry')
async def retry_document(
        document_id: int,
        user_info: dict = Depends(get_current_user),
        db: AsyncSession = Depends(get_db)):

    document = await get_document_by_id(document_id, db)

    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    if document.user_id != user_info['user_id']:
        raise HTTPException(status_code=403, detail="无权限操作该文档")
    if document.status != "failed":
        raise HTTPException(status_code=409, detail="只有解析失败的文档可以重试")

    try:
        await bm25_cache.invalidate(
            user_id=user_info["user_id"],
            document_id=document.id,
        )
        await delete_document_chunks(document.id, db)
        await delete_vectors(document.id)
        document.status = "processing"
        document.error_message = None
        document.chunk_count = 0
        await db.commit()

        await ingest_document(user_info["user_id"],document, db)
        await db.commit()
        await db.refresh(document)

        return {
            "id": document.id,
            "status": document.status,
            "chunk_count": document.chunk_count,
            "error_message": document.error_message,
        }

    except Exception as exc:
        await db.rollback()

        failed_document = await get_document_by_id(document_id, db)
        if failed_document:
            failed_document.status = "failed"
            failed_document.error_message = str(exc)
            failed_document.chunk_count = 0
            await db.commit()

        raise HTTPException(
            status_code=422,
            detail="文档重新解析失败",
        ) from exc



@router.get('/documents')
async def get_documents(user_info=Depends(get_current_user),db: AsyncSession=Depends(get_db)):
    documents_list = await get_all_documents_by_user(user_info['user_id'],db)
    documents = [
        {
            'id':document.id,
            'filename':document.filename,
            'content_type':document.content_type,
            'status':document.status,
            'file_hash':document.file_hash,
            'error_message':document.error_message,
            'chunk_count':document.chunk_count
        }
        for document in documents_list
    ]
    return JSONResponse(status_code=200,content=documents)



@router.delete('/documents/{document_id}')
async def delete_document(document_id: int,user_info=Depends(get_current_user),db: AsyncSession=Depends(get_db)):
    document = await get_document_by_id(document_id,db)
    if not document:
        raise HTTPException(status_code=404,detail='文档不存在')
    if document.user_id != user_info['user_id']:
        raise HTTPException(status_code=403, detail='无权限删除该文档')
    await db.delete(document)
    try:
        await delete_file(document.file_path)
        await delete_vectors(document_id)
        await bm25_cache.invalidate(
            user_id=user_info["user_id"],
            document_id=document.id,
        )
    except:
        raise HTTPException(status_code=500,detail='文件删除失败')

    return JSONResponse(status_code=200,content={'message':'删除成功'})


@router.post('/query',response_model=QueryResponse)
async def query(
        request_app: Request,
        request: QueryRequest,
        user_info=Depends(get_current_user),
        db: AsyncSession=Depends(get_db)):
    return await ask_rag(
        rag_graph=request_app.app.state.rag_graph,
        db=db,
        user_info=user_info,
        request=request,
    )



