from langgraph.constants import START, END
from langgraph.graph import StateGraph

from apps.rag.workflow.context import RAGRuntimeContext
from apps.rag.workflow.nodes import retrieve_node, assess_node, rewrite_node, generate_node, reject_node, \
    route_after_assess
from apps.rag.workflow.state import RAGState

builder = StateGraph(RAGState,context_schema=RAGRuntimeContext,)

builder.add_node("retrieve", retrieve_node)
builder.add_node("assess", assess_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)
builder.add_node("reject", reject_node)


builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "assess")
builder.add_conditional_edges(
    "assess",
    route_after_assess,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "reject": "reject",
    },
)
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)
builder.add_edge("reject", END)

rag_graph = builder.compile()