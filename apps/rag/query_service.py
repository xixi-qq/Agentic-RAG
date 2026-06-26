from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from apps.conversations.crud import (
    add_message,
    create_conversation,
    get_conversation_by_id,
    update_conversation_title,
)
from apps.conversations.service import generate_conversation_title
from apps.rag.schemas import QueryRequest, QueryResponse
from apps.rag.service import organize_response
from apps.rag.workflow.context import RAGRuntimeContext, RetrievalConfig


FALLBACK_ANSWER = "未在文档中找到足够信息，无法回答"


async def get_or_create_query_conversation(
    *,
    db: AsyncSession,
    user_id: int,
    conversation_id: str | None,
) -> tuple[str, bool]:
    if conversation_id is None:
        conversation = await create_conversation(db, user_id)
        return conversation.id, True

    conversation = await get_conversation_by_id(
        db,
        user_id,
        conversation_id,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    return conversation.id, False


async def invoke_rag_graph(
    *,
    rag_graph,
    db: AsyncSession,
    user_id: int,
    conversation_id: str,
    request: QueryRequest,
) -> dict:
    return await rag_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request.user_query,
                }
            ],
            "original_query": request.user_query,
            "search_query": request.user_query,
            "rewrite_count": 0,
        },
        config={
            "configurable": {
                "thread_id": conversation_id,
            },
        },
        context=RAGRuntimeContext(
            user_id=user_id,
            document_id=request.document_id,
            db=db,
            retrieval=RetrievalConfig(
                candidate_k=request.candidate_k,
                final_k=request.final_k,
                score_threshold=request.score_threshold,
            ),
        ),
    )


async def persist_query_messages(
    *,
    db: AsyncSession,
    user_id: int,
    conversation_id: str,
    user_query: str,
    answer: str,
    should_generate_title: bool,
) -> None:
    await add_message(db, user_id, conversation_id, "user", user_query)
    await add_message(db, user_id, conversation_id, "assistant", answer)

    if should_generate_title:
        title = await generate_conversation_title(user_query, answer)
        await update_conversation_title(
            db,
            user_id,
            conversation_id,
            title,
        )


async def ask_rag(
    *,
    rag_graph,
    db: AsyncSession,
    user_info: dict,
    request: QueryRequest,
) -> QueryResponse:
    user_id = user_info["user_id"]
    conversation_id, is_new_conversation = await get_or_create_query_conversation(
        db=db,
        user_id=user_id,
        conversation_id=request.conversation_id,
    )

    result = await invoke_rag_graph(
        rag_graph=rag_graph,
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        request=request,
    )

    answer = result.get("answer", FALLBACK_ANSWER)
    await persist_query_messages(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        user_query=request.user_query,
        answer=answer,
        should_generate_title=is_new_conversation,
    )

    candidates = result.get("candidates", [])
    return organize_response(conversation_id, answer, candidates)
