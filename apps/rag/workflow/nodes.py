import logging

from langgraph.runtime import Runtime

from apps.rag.agent import get_content
from apps.rag.fusion import retrieve_and_rerank
from apps.rag.schemas import RouteDecision, AssessDecision
from apps.rag.workflow.context import RAGRuntimeContext
from apps.rag.workflow.prompts import router_prompt, contextualize_prompt, rewrite_prompt, rag_prompt, assess_prompt, \
    chat_prompt
from apps.rag.workflow.state import RAGState
from config.agent_config import model

logger = logging.getLogger(__name__)


async def router_query(state: RAGState, runtime: Runtime[RAGRuntimeContext]):
    system_prompt = router_prompt
    router_model = model.with_structured_output(RouteDecision)
    response = await router_model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"]])
    logger.info(
        "rag.router decision=%s user_id=%s document_id=%s",
        response.decision,
        runtime.context.user_id,
        runtime.context.document_id,
    )
    return {"need_retrieve" : response.decision}

async def chat_node(state: RAGState, runtime: Runtime[RAGRuntimeContext]):
    system_prompt = chat_prompt
    answer = await model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"]])
    logger.info(
        "rag.chat answer_length=%s user_id=%s document_id=%s",
        len(answer.content),
        runtime.context.user_id,
        runtime.context.document_id,
    )
    return {
        "answer": answer.content,
        "candidates": [],
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
    search_query = response.content.strip()
    logger.info(
        "rag.contextualize search_query=%r user_id=%s document_id=%s",
        search_query[:100],
        runtime.context.user_id,
        runtime.context.document_id,
    )
    return {"search_query": search_query}


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
    logger.info(
        "rag.retrieve query=%r candidate_count=%s user_id=%s document_id=%s",
        query[:100],
        len(candidates),
        runtime.context.user_id,
        runtime.context.document_id,
    )
    return {
        "search_query": query,
        "candidates": candidates,
        "retrieved_content": content
    }


async def assess_node(
    state: RAGState,
    runtime: Runtime[RAGRuntimeContext],
) -> dict:
    assess_model = model.with_structured_output(AssessDecision)
    system_prompt = assess_prompt
    res = await assess_model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"],
                               {"role": "user", "content":state["retrieved_content"]}])

    logger.info(
        "rag.assess decision=%s rewrite_count=%s candidate_count=%s user_id=%s document_id=%s",
        res.decision,
        state.get("rewrite_count", 0),
        len(state.get("candidates", [])),
        runtime.context.user_id,
        runtime.context.document_id,
    )
    return {
        "retrieval_sufficient": res.decision,
    }




async def rewrite_node(
    state: RAGState,
    runtime: Runtime[RAGRuntimeContext],
) -> dict:
    system_prompt = rewrite_prompt

    response = await model.ainvoke([{"role": "system", "content":system_prompt},
                                   *state["messages"]])

    rewritten_query = response.content.strip()

    logger.info(
        "rag.rewrite search_query=%r rewrite_count=%s user_id=%s document_id=%s",
        rewritten_query[:100],
        state.get("rewrite_count", 0) + 1,
        runtime.context.user_id,
        runtime.context.document_id,
    )
    return {
        "search_query": rewritten_query,
        "rewrite_count": (
            state.get("rewrite_count", 0) + 1
        ),
    }


async def generate_node(
    state: RAGState,
    runtime: Runtime[RAGRuntimeContext],
) -> dict:
    system_prompt = rag_prompt
    answer = await model.ainvoke([{"role": "system", "content":system_prompt},
                                 *state["messages"],
                                  {"role": "user","content": state["retrieved_content"]},])

    logger.info(
        "rag.generate answer_length=%s source_count=%s user_id=%s document_id=%s",
        len(answer.content),
        len(state.get("candidates", [])),
        runtime.context.user_id,
        runtime.context.document_id,
    )
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
    runtime: Runtime[RAGRuntimeContext],
) -> dict:
    logger.info(
        "rag.reject rewrite_count=%s candidate_count=%s user_id=%s document_id=%s",
        state.get("rewrite_count", 0),
        len(state.get("candidates", [])),
        runtime.context.user_id,
        runtime.context.document_id,
    )
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
