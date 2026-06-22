import re

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.rag.schemas import RetrieveItem, RetrieveItemMetadata
from models.documents import Document, DocumentChunk


TOKEN_PATTERN = re.compile(
    r"[a-zA-Z][a-zA-Z0-9_.+-]*|"
    r"[\u4e00-\u9fff]+|"
    r"\d+(?:\.\d+)*"
)


def tokenize(text: str) -> list[str]:
    """兼顾中文分词和 thread_id、ToolStrategy 等技术术语。"""
    tokens = []

    for part in TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(
                token.strip()
                for token in jieba.cut(part)
                if token.strip()
            )
        else:
            tokens.append(part)

    return tokens


async def get_bm25_chunks(
    user_id: int,
    document_id: int | None,
    db: AsyncSession,
):
    stmt = (
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.user_id == user_id,
            Document.status == "completed",
        )
        .order_by(DocumentChunk.chunk_index)
    )

    if document_id is not None:
        stmt = stmt.where(Document.id == document_id)

    result = await db.execute(stmt)
    return result.all()


async def search_bm25(
    query: str,
    user_id: int,
    document_id: int | None,
    db: AsyncSession,
    top_k: int = 15,
) -> list[RetrieveItem]:
    rows = await get_bm25_chunks(
        user_id=user_id,
        document_id=document_id,
        db=db,
    )

    if not rows:
        return []

    tokenized_corpus = [
        tokenize(chunk.content)
        for chunk, _filename in rows
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []

    for index in ranked_indexes:
        score = float(scores[index])

        # 没有任何关键词匹配
        if score <= 0:
            break

        chunk, filename = rows[index]

        results.append(
            RetrieveItem(
                content=chunk.content,
                score=score,
                metadata=RetrieveItemMetadata(
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    filename=filename,
                ),
            )
        )
    return results


