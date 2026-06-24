from langgraph.constants import START, END
from langgraph.graph import StateGraph

from apps.rag.workflow.context import RAGRuntimeContext
from apps.rag.workflow.nodes import retrieve_node, assess_node, rewrite_node, generate_node, reject_node, \
    router_after_assess, router_query, contextualize_node, router_after_router, router_after_retrieve, chat_node
from apps.rag.workflow.state import RAGState


def create_rag_graph(checkpoint):
    builder = StateGraph(RAGState,context_schema=RAGRuntimeContext,)

    builder.add_node("router", router_query)
    builder.add_node("chat", chat_node)
    builder.add_node("contextualize", contextualize_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("assess", assess_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)
    builder.add_node("reject", reject_node)


    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        router_after_router,
        {
            "chat": "chat",
            "contextualize": "contextualize",
        },
    )
    builder.add_edge("contextualize", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        router_after_retrieve,
        {
            "assess": "assess",
            "reject": "reject",
        },
    )
    builder.add_conditional_edges(
        "assess",
        router_after_assess,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "reject": "reject",
        },
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("chat", END)
    builder.add_edge("reject", END)

    return builder.compile(checkpointer=checkpoint,)