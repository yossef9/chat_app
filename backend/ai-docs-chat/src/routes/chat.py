from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from src.auth import get_current_active_user
from src.models.user import User
from src.models.chat import ChatOut, ChatList
from src.database import get_db
from src.services.rag_service import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatAskRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None


@router.post("/ask")
async def ask_rag(req: ChatAskRequest, current_user: User = Depends(get_current_active_user)):
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Use server's API key from environment
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Server Groq API key not configured")

    result = await rag_service.query_question(
        current_user,
        req.question,
        api_key,
        req.document_ids,
    )
    return result


@router.get("/history", response_model=ChatList)
async def get_chat_history(
    current_user: User = Depends(get_current_active_user),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return")
):
    """Get chat history for the current user"""
    db = get_db()
    
    # Query chat history for the current user
    cursor = db["chat_history"].find(
        {"user_id": str(current_user.id)}
    ).sort("created_at", -1).skip(skip).limit(limit)
    
    chats = []
    async for chat_doc in cursor:
        chats.append(ChatOut(
            id=chat_doc["id"],
            user_id=chat_doc["user_id"],
            search_message=chat_doc["search_message"],
            response=chat_doc["response"],
            created_at=chat_doc["created_at"],
            updated_at=chat_doc.get("updated_at")
        ))
    
    # Get total count
    total = await db["chat_history"].count_documents({"user_id": str(current_user.id)})
    
    return ChatList(chats=chats, total=total)


@router.get("/history/{chat_id}", response_model=ChatOut)
async def get_chat_by_id(
    chat_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific chat message by ID"""
    db = get_db()
    
    chat_doc = await db["chat_history"].find_one({
        "id": chat_id,
        "user_id": str(current_user.id)
    })
    
    if not chat_doc:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return ChatOut(
        id=chat_doc["id"],
        user_id=chat_doc["user_id"],
        search_message=chat_doc["search_message"],
        response=chat_doc["response"],
        created_at=chat_doc["created_at"],
        updated_at=chat_doc.get("updated_at")
    )


@router.delete("/history/{chat_id}")
async def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a specific chat message by ID"""
    db = get_db()
    
    result = await db["chat_history"].delete_one({
        "id": chat_id,
        "user_id": str(current_user.id)
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return {"message": "Chat deleted successfully"}


@router.delete("/history")
async def delete_all_chat_history(
    current_user: User = Depends(get_current_active_user)
):
    """Delete all chat history for the current user"""
    db = get_db()
    
    result = await db["chat_history"].delete_many({
        "user_id": str(current_user.id)
    })
    
    return {"message": f"Deleted {result.deleted_count} chat messages"}


