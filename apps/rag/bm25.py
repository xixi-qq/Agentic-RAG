import re

import jieba
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.rag.schemas import RetrieveItem, RetrieveItemMetadata
from models.documents import Document, DocumentChunk
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from rank_bm25 import BM25Okapi




CacheKey = tuple[int, int | None]


@dataclass
class BM25CacheEntry:
    bm25: BM25Okapi
    rows: list
    chunk_count: int


class BM25IndexCache:
    def __init__(self, max_chunks: int = 5000):
        self.max_chunks = max_chunks

        self._entries: OrderedDict[
            CacheKey,
            BM25CacheEntry,
        ] = OrderedDict()

        self._total_chunks = 0
        self._lock = asyncio.Lock()

    async def get(self,key: CacheKey) -> BM25CacheEntry:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry

    async def put(
        self,
        key: CacheKey,
        entry: BM25CacheEntry,
    ) -> None:
        async with self._lock:
            old_entry = self._entries.pop(key, None)

            if old_entry is not None:
                self._total_chunks -= old_entry.chunk_count

            # 单个索引已经超过全部容量时，不缓存
            if entry.chunk_count > self.max_chunks:
                return

            self._entries[key] = entry
            self._total_chunks += entry.chunk_count

            # 删除最久没有使用的索引
            while self._total_chunks > self.max_chunks:
                _old_key, removed = self._entries.popitem(
                    last=False,
                )
                self._total_chunks -= removed.chunk_count

    async def invalidate(
        self,
        user_id: int,
        document_id: int | None = None,
    ) -> None:
        async with self._lock:
            keys_to_remove = []

            for key in self._entries:
                cached_user_id, cached_document_id = key

                if cached_user_id != user_id:
                    continue

                # document_id=None表示清理该用户全部索引
                if (
                    document_id is None
                    or cached_document_id is None
                    or cached_document_id == document_id
                ):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                entry = self._entries.pop(key)
                self._total_chunks -= entry.chunk_count

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._total_chunks = 0

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    @property
    def entry_count(self) -> int:
        return len(self._entries)

bm25_cache = BM25IndexCache()

async def build_bm25_entry(
    user_id: int,
    document_id: int | None,
    db: AsyncSession,
) -> BM25CacheEntry | None:
    rows = await get_bm25_chunks(
        user_id=user_id,
        document_id=document_id,
        db=db,
    )

    if not rows:
        return None

    tokenized_corpus = [
        tokenize(chunk.content)
        for chunk, _filename in rows
    ]

    return BM25CacheEntry(
        bm25=BM25Okapi(tokenized_corpus),
        rows=rows,
        chunk_count=len(rows),
    )

async def get_or_build_bm25_entry(
    user_id: int,
    document_id: int | None,
    db: AsyncSession,
) -> BM25CacheEntry | None:
    key = (user_id, document_id)

    cached = await bm25_cache.get(key)

    if cached is not None:
        return cached

    entry = await build_bm25_entry(
        user_id=user_id,
        document_id=document_id,
        db=db,
    )

    if entry is not None:
        await bm25_cache.put(key, entry)

    return entry





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
    entry = await get_or_build_bm25_entry(
        user_id=user_id,
        document_id=document_id,
        db=db,
    )

    if entry is None:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = entry.bm25.get_scores(query_tokens)

    ranked_indexes = sorted(range(len(entry.rows)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []

    for index in ranked_indexes:
        score = float(scores[index])

        # 没有任何关键词匹配
        if score <= 0:
            break

        chunk, filename = entry.rows[index]

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


