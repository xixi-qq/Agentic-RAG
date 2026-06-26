import asyncio
import hashlib

from fastapi import UploadFile
from apps.rag.schemas import ParsedPage
from models.documents import Document
from settings import HASH_CHUNK_SIZE


TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gbk")


async def calculate_file_hash(file: UploadFile) -> str:
    sha256 = hashlib.sha256()
    await file.seek(0)
    while chunk := await file.read(HASH_CHUNK_SIZE):
        sha256.update(chunk)
    await file.seek(0)
    return sha256.hexdigest()


import fitz


def extract_page_text_with_layout(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")

    text_blocks = []
    for block in blocks:
        x0, y0, x1, y1, text, *_ = block
        text = text.strip()
        if not text:
            continue
        text_blocks.append((x0, y0, x1, y1, text))

    if not text_blocks:
        return ""

    page_width = page.rect.width
    mid_x = page_width / 2

    left_blocks = []
    right_blocks = []

    for block in text_blocks:
        x0, y0, x1, y1, text = block
        center_x = (x0 + x1) / 2

        if center_x < mid_x:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    # 简单判断：左右两侧都有一定数量文本块，就按双栏处理
    is_two_column = len(left_blocks) >= 3 and len(right_blocks) >= 3

    if is_two_column:
        ordered = sorted(left_blocks, key=lambda b: (b[1], b[0]))
        ordered += sorted(right_blocks, key=lambda b: (b[1], b[0]))
    else:
        ordered = sorted(text_blocks, key=lambda b: (b[1], b[0]))

    return "\n\n".join(block[4] for block in ordered)

def parse_pdf_sync(user_id: int, file_path: str, document: Document):
    pdf = fitz.open(file_path)
    pages = []

    for index, page in enumerate(pdf):
        text = extract_page_text_with_layout(page)

        if not text.strip():
            continue

        pages.append(
            ParsedPage(
                user_id=user_id,
                page_number=index + 1,
                content=text,
                document_id=document.id,
                filename=document.filename,
            )
        )

    if not pages:
        raise ValueError("PDF 未提取到有效文本，可能是扫描版 PDF，需要 OCR")

    return pages


async def parse_pdf(
    user_id: int,
    file_path: str,
    document: Document,
) -> list[ParsedPage]:
    return await asyncio.to_thread(parse_pdf_sync, user_id, file_path, document)


def read_text_with_fallback(file_path: str) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("文本文件编码不支持")


def parse_text_sync(
    user_id: int,
    file_path: str,
    document: Document,
) -> list[ParsedPage]:
    text = read_text_with_fallback(file_path)
    if not text.strip():
        raise ValueError("文本未提取到有效文本")

    return [
        ParsedPage(
            user_id=user_id,
            page_number=1,
            content=text,
            document_id=document.id,
            filename=document.filename,
        )
    ]


async def parse_text(
    user_id: int,
    file_path: str,
    document: Document,
) -> list[ParsedPage]:
    return await asyncio.to_thread(parse_text_sync, user_id, file_path, document)


async def parse_document(
    user_id: int,
    path: str,
    document: Document,
) -> list[ParsedPage]:
    if document.content_type == "application/pdf":
        return await parse_pdf(user_id, path, document)
    if document.content_type in ("text/plain", "text/markdown"):
        return await parse_text(user_id, path, document)
    raise ValueError("不支持的文件类型")
