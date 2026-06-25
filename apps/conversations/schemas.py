from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ConversationItem(BaseModel):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationItem]


class ConversationMessageItem(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    updated_at: datetime


class ConversationDetailResponse(BaseModel):
    id: str
    title: str | None
    messages: list[ConversationMessageItem]
    created_at: datetime
    updated_at: datetime



class DeleteConversationResponse(BaseModel):
    message: str