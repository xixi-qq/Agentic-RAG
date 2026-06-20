import asyncio
import hashlib

from fastapi import UploadFile
from pypdf import PdfReader
from apps.rag.schemas import ParsedPage
from models.documents import Document
from settings import HASH_CHUNK_SIZE


async def calculate_file_hash(file: UploadFile) -> str:
    sha256 = hashlib.sha256()
    await file.seek(0)
    while chunk := await file.read(HASH_CHUNK_SIZE):
        sha256.update(chunk)
    await file.seek(0)
    return sha256.hexdigest()


# 同步
def parse_pdf_sync(user_id: int,file_path: str,document: Document) -> list[ParsedPage]:

    pdf_reader = PdfReader(file_path)
    pages = []
    for index,page in enumerate(pdf_reader.pages):
        text = page.extract_text() or ""
        if text.strip() == "":
            continue
        pages.append(ParsedPage(user_id=user_id,page_number=index+1,content=text,document_id=document.id,filename=document.filename))
    if not pages:
        raise ValueError("PDF未提取到有效文本")
    return pages

# 异步
async def parse_pdf(user_id: int,file_path: str,document: Document) -> list[ParsedPage]:
    return await asyncio.to_thread(parse_pdf_sync,user_id,file_path,document)


def parse_text_sync(user_id: int,file_path: str,document: Document) -> list[ParsedPage]:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    if text.strip() == "":
        raise ValueError("文本未提取到有效文本")
    return  [ParsedPage(user_id=user_id,page_number=1,content=text,document_id=document.id,filename=document.filename)]



async def parse_text(user_id: int,file_path: str,document: Document) -> list[ParsedPage]:
    return await asyncio.to_thread(parse_text_sync,user_id,file_path,document)



async def parse_document(user_id,path: str, document: Document) -> list[ParsedPage]:

    if document.content_type == "application/pdf":
        return await parse_pdf(user_id,path,document)
    elif document.content_type in ("text/plain","text/markdown"):
        return await parse_text(user_id,path,document)
    else:
        raise ValueError("不支持的文件类型")