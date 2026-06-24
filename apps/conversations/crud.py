from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.conversations import Conversation, ConversationMessage


async def create_conversation(db: AsyncSession,
                              user_id: int,
                              title: str | None = None) -> Conversation:

    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    await db.flush()
    await db.refresh(conversation)
    return conversation


async def get_all_conversation_titles(db: AsyncSession,
                           user_id: int) -> list:
    stmt = select(Conversation.title).where(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at)
    result = await db.scalars(stmt)
    return result.all()


async def get_conversation_by_id(db: AsyncSession,
                                 user_id: int,
                                 conversation_id: str) -> Conversation | None:
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,Conversation.user_id == user_id
    )
    result = await db.execute(stmt)
    return result.scalars().first()


async def delete_conversation(db: AsyncSession,
                              user_id: int,
                              conversation_id: str) -> bool:

    conversation = await get_conversation_by_id(db, user_id,conversation_id)
    if conversation is None:
        return False
    await db.delete(conversation)
    await db.flush()
    return True

MessageRole = Literal["user", "assistant"]


async def add_message(db: AsyncSession,
                        user_id: int,
                        conversation_id: str,
                        role: MessageRole,
                        content: str) -> ConversationMessage:
    if role not in {"user", "assistant"}:
        raise ValueError("不支持的消息角色")

    conversation = await get_conversation_by_id(
        db,
        user_id,
        conversation_id,
    )

    if conversation is None:
        raise ValueError("会话不存在或无权限")
    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError("消息内容不能为空")

    message = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=normalized_content,
    )
    db.add(message)
    await db.flush()
    await db.refresh(message)
    return message


async def get_messages(
    db: AsyncSession,
    user_id: int,
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[ConversationMessage]:
    conversation = await get_conversation_by_id(
        db,
        user_id,
        conversation_id,
    )

    if conversation is None:
        raise ValueError("会话不存在或无权限")
    messages = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id,)
        .order_by(ConversationMessage.id.desc(),)
        .limit(limit)
        .offset(offset),
    )
    res = list(messages.scalars().all())
    res.reverse()
    return res
