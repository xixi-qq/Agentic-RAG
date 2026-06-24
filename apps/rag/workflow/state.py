from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated, Literal

from apps.rag.schemas import RetrieveItem


class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    original_query: str
    need_retrieve: Literal["need","unneeded"]
    search_query: str
    candidates: list[RetrieveItem]
    retrieved_content : str
    retrieval_sufficient: str
    rewrite_count: int
    answer: str