from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from apps.conversations.crud import get_all_conversations, get_conversation_by_id, get_messages, delete_conversation_by_id
from apps.conversations.schemas import ConversationListResponse, ConversationItem, ConversationDetailResponse, \
    ConversationMessageItem, DeleteConversationResponse
from config.db_config import get_db
from utils.jwt import get_current_user

router = APIRouter(prefix='/conversations',tags=['conversation'])


@router.get('/list',response_model=ConversationListResponse)
async def get_conversations(user_info=Depends(get_current_user),
                            db: AsyncSession=Depends(get_db)):

    conversations_list= await get_all_conversations(db,user_info['user_id'])
    conversations = [
            ConversationItem(id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at)
         for conversation in conversations_list
    ]
    return ConversationListResponse(conversations=conversations)

@router.get('/{conversation_id}/messages',response_model=ConversationDetailResponse)
async def get_conversation(conversation_id: str = Path(...,description="会话id"),
                           offset: int = Query(0,description="跳过数量"),
                           limit: int = Query(50,description="限制数量"),
                           user_info=Depends(get_current_user),
                           db: AsyncSession=Depends(get_db)):

    conversation = await get_conversation_by_id(db,user_info['user_id'],conversation_id)
    if not conversation:
        raise HTTPException(status_code=404,detail='会话不存在')
    messages_obj = await get_messages(db,user_info["user_id"],conversation_id,offset,limit)
    messages = [
            ConversationMessageItem(id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            updated_at=message.updated_at)
            for message in messages_obj
    ]
    return ConversationDetailResponse(id=conversation.id,
                                      title=conversation.title,
                                      messages=messages,
                                      created_at=conversation.created_at,
                                      updated_at=conversation.updated_at)


@router.delete('/{conversation_id}',response_model=DeleteConversationResponse)
async def delete_conversation(conversation_id: str = Path(...,description="会话id"),
                           user_info=Depends(get_current_user),
                           db: AsyncSession=Depends(get_db)):

    res = await delete_conversation_by_id(db,user_info['user_id'],conversation_id)
    if res:
        return DeleteConversationResponse(message='删除成功')
    raise HTTPException(status_code=404,detail='会话不存在')