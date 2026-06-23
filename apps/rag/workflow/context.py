from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession


from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_k: int = 30
    final_k: int = 5
    score_threshold: float = 0.0


@dataclass
class RAGRuntimeContext:
    user_id: int
    document_id: int | None
    db: AsyncSession
    retrieval: RetrievalConfig