import tempfile
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, HTTPException
from starlette.concurrency import run_in_threadpool
from config.minio_config import MINIO_BUCKET, minio_client
from settings import SAFE_FILE_SIZE


def make_object_name(user_id: int, filename: str) -> str:
    safe_filename = filename.replace("\\", "_").replace("/", "_")
    return f"users/{user_id}/{uuid4().hex}/{safe_filename}"


async def  download_to_temp(object_name: str,suffix: str) -> Path:
    """
    从minio下载文件到本地临时文件,并返回该临时文件的路径
    :param object_name:
    :param suffix:
    :return:
    """
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp.name)
    temp.close()
    try:
        await run_in_threadpool(
            minio_client.fget_object,
            MINIO_BUCKET,
            object_name,
            str(temp_path),
        )
        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


async def upload_file(
    user_id: int,
    file: UploadFile,
) -> tuple[str, int]:
    """
    把文件上传到minio
    :param user_id:
    :param file:
    :return:
    """

    object_name = make_object_name(
        user_id,
        file.filename or "unknown",
    )

    # 获取 SpooledTemporaryFile 的实际长度
    await run_in_threadpool(file.file.seek, 0, 2)
    file_size = file.file.tell()
    if file_size > SAFE_FILE_SIZE:
        raise HTTPException(status_code=413,detail="文件大小不能超过 20MB")
    await run_in_threadpool(file.file.seek, 0)

    await run_in_threadpool(
        minio_client.put_object,
        MINIO_BUCKET,
        object_name,
        file.file,
        file_size,
        content_type=file.content_type,
    )

    return object_name, file_size


async def delete_file(object_name: str) -> None:
    """
    删除minio内的文件
    :param object_name:
    :return:
    """
    await run_in_threadpool(
        minio_client.remove_object,
        MINIO_BUCKET,
        object_name,
    )


async def get_download_url(object_name: str) -> str:
    """
    获取临时下载的链接,10分钟后过期
    :param object_name:
    :return: path: str
    """
    return await run_in_threadpool(
        minio_client.presigned_get_object,
        MINIO_BUCKET,
        object_name,
        expires=timedelta(minutes=10),
    )



