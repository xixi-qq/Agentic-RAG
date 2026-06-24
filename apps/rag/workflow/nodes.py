from langgraph.runtime import Runtime

from apps.rag.agent import get_content
from apps.rag.fusion import retrieve_and_rerank
from apps.rag.schemas import RouteDecision, AssessDecision
from apps.rag.workflow.context import RAGRuntimeContext
from apps.rag.workflow.prompts import router_prompt, contextualize_prompt, rewrite_prompt, rag_prompt, assess_prompt, \
    chat_prompt
from apps.rag.workflow.state import RAGState
from config.agent_config import model


async def router_query(state: RAGState, runtime: Runtime[RAGRuntimeContext]):
    system_prompt = router_prompt
    router_model = model.with_structured_output(RouteDecision)
    response = await router_model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"]])
    return {"need_retrieve" : response.decision}

async def chat_node(state: RAGState, runtime: Runtime[RAGRuntimeContext]):
    system_prompt = chat_prompt
    answer = await model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"]])
    return {
        "answer": answer.content,
    "messages": [
        {
            "role": "assistant",
            "content": answer.content,
        }
    ],
}


async def contextualize_node(state: RAGState, runtime: Runtime[RAGRuntimeContext]):
    system_prompt = contextualize_prompt
    response = await model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"]])
    return {"search_query": response.content.strip()}


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
    content = get_content(candidates)
    return {
        "search_query": query,
        "candidates": candidates,
        "retrieved_content": content
    }


async def assess_node(
    state: RAGState,
) -> dict:
    assess_model = model.with_structured_output(AssessDecision)
    system_prompt = assess_prompt
    res = await assess_model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"],
                               {"role": "user", "content":state["retrieved_content"]}])

    return {
        "retrieval_sufficient": res.decision,
    }




async def rewrite_node(
    state: RAGState,
) -> dict:
    system_prompt = rewrite_prompt

    response = await model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"]])

    rewritten_query = response.content.strip()

    return {
        "search_query": rewritten_query,
        "rewrite_count": (
            state.get("rewrite_count", 0) + 1
        ),
    }


async def generate_node(
    state: RAGState,
) -> dict:
    system_prompt = rag_prompt
    answer = await model.ainvoke([{"role": "system", "content":system_prompt},
                                 *state["messages"],
                                  {"role": "user","content": state["retrieved_content"]},])

    return {
    "answer": answer.content,
    "messages": [
        {
            "role": "assistant",
            "content": answer.content.strip(),
        }
    ],
}


async def reject_node(
    state: RAGState,
) -> dict:
    return {
        "answer": "未在文档中找到足够信息，无法回答",
        "candidates": [],
        "messages": [
            {
                "role": "assistant",
                "content": "未在文档中找到足够信息，无法回答",
            }
        ],
    }



def router_after_router(state: RAGState) -> str:
    if state["need_retrieve"] == "need":
        return "contextualize"
    return "chat"

def router_after_retrieve(state: RAGState) -> str:
    if state["candidates"]:
        return "assess"
    return "reject"

def router_after_assess(state: RAGState) -> str:
    if state["retrieval_sufficient"] == "sufficient":
        return "generate"
    if state.get("rewrite_count", 0) < 1:
        return "rewrite"
    return "reject"