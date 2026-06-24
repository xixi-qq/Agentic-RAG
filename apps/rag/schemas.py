from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ParsedPage(BaseModel):
    user_id: int
    document_id: int
    content: str
    page_number: int | None
    filename: str



class Chunk(BaseModel):
    user_id: int
    content: str
    chunk_index: int
    page_number: int | None
    document_id: int
    vector_id: str | None = None
    filename: str


class Vector(BaseModel):
    user_id: int
    vector_id: str
    page_number: int | None
    document_id: int
    chunk_index: int
    vector: list[float]
    filename: str

class RetrieveItemMetadata(BaseModel):
    document_id: int
    page_number: int | None
    chunk_index: int
    filename: str

class RetrieveItem(BaseModel):
    content: str
    score: float
    metadata: RetrieveItemMetadata


class QueryRequest(BaseModel):
    user_query: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    document_id: int | None = None
    candidate_k: int = Field(default=30, ge=1, le=50)
    final_k: int = Field(default=5, ge=1, le=10)
    score_threshold: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_k(self):
        if self.final_k > self.candidate_k:
            raise ValueError("final_k 不能大于 candidate_k")
        return self



class QuerySource(BaseModel):
    filename: str
    page_number: int | None
    score: float


class QueryResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[QuerySource]



class RouteDecision(BaseModel):
    decision: Literal["need", "unneeded"] = Field(
        description="need 表示需要检索知识库，unneeded 表示不需要检索"
    )


class AssessDecision(BaseModel):
    decision: Literal["sufficient", "insufficient"] = Field(
        description="sufficient 表示检索上下文足够回答，insufficient 表示不足以回答"
    )
