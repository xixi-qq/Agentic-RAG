from langgraph.runtime import Runtime

from apps.rag.fusion import retrieve_and_rerank
from apps.rag.workflow.context import RAGRuntimeContext
from apps.rag.workflow.state import RAGState


async def retrieve_node(state: RAGState, runtime: Runtime[RAGRuntimeContext]):
    query = state.get("search_query") or state["original_query"]
    config = runtime.context.retrieval
    candidates = await retrieve_and_rerank(
    query=query,
    user_id=runtime.context.user_id,
    document_id=runtime.context.document_id,
    db=runtime.context.db,
    candidate_k=config.candidate_k,
    final_k=config.final_k,
    score_threshold=config.score_threshold,
)
    return {
        "search_query": query,
        "candidates": candidates,
    }


async def assess_node(
    state: RAGState,
) -> dict:
    candidates = state.get("candidates", [])

    return {
        "retrieval_sufficient": bool(candidates),
    }

from config.agent_config import model


async def rewrite_node(
    state: RAGState,
) -> dict:
    prompt = f"""
请将用户问题改写成一个完整、明确、适合知识库检索的问题。

要求：
- 不要回答问题
- 不要添加未知事实
- 保留技术术语
- 只输出一个改写后的问题

用户问题：
{state["original_query"]}
"""

    response = await model.ainvoke(prompt)

    rewritten_query = response.content.strip()

    return {
        "search_query": rewritten_query,
        "rewrite_count": (
            state.get("rewrite_count", 0) + 1
        ),
    }


from apps.rag.agent import response_user_query


async def generate_node(
    state: RAGState,
) -> dict:
    answer = await response_user_query(
        state["original_query"],
        state["candidates"],
    )

    return {"answer": answer}


async def reject_node(
    state: RAGState,
) -> dict:
    return {
        "answer": "未在文档中找到足够信息，无法回答",
        "candidates": [],
    }


def route_after_assess(
    state: RAGState,
) -> str:
    if state["retrieval_sufficient"]:
        return "generate"

    if state.get("rewrite_count", 0) < 1:
        return "rewrite"

    return "reject"