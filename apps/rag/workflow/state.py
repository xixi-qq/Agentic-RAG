from typing_extensions import TypedDict

from apps.rag.schemas import RetrieveItem


class RAGState(TypedDict, total=False):
    messages: list
    original_query: str
    search_query: str
    candidates: list[RetrieveItem]
    retrieval_sufficient: bool
    rewrite_count: int
    answer: str