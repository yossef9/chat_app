# src/routes/search_history.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime
import uuid

from src.database import get_db
from src.models.search_history import SearchHistory, ChatSession, ChatMessage, CreateSessionRequest
from src.auth import get_current_active_user
from src.services.rag_service import rag_service

router = APIRouter(prefix="/chat/sessions", tags=["chat-sessions"])

# Create a search history entry
@router.post("", response_model=ChatSession)
async def create_session(
    request: CreateSessionRequest,
    current_user=Depends(get_current_active_user)
):
    db = get_db()
    session = ChatSession(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        name=request.name,
        document_ids=request.document_ids or [],
    )
    await db["chat_sessions"].insert_one(session.dict())
    return session


# Get all search history for the current user
@router.get("", response_model=List[ChatSession])
async def list_sessions(current_user=Depends(get_current_active_user)):
    db = get_db()
    cur = db["chat_sessions"].find({"user_id": current_user.id}).sort("updated_at", -1)
    items = await cur.to_list(100)
    return [ChatSession(**i) for i in items]

# Get a specific search history entry by id
@router.get("/{session_id}", response_model=ChatSession)
async def get_session(session_id: str, current_user=Depends(get_current_active_user)):
    db = get_db()
    doc = await db["chat_sessions"].find_one({"id": session_id, "user_id": current_user.id})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    return ChatSession(**doc)

# Get all chat messages for a specific session
@router.get("/{session_id}/messages", response_model=List[ChatMessage])
async def get_chat_messages(session_id: str, current_user=Depends(get_current_active_user)):
    db = get_db()
    
    # Verify session exists and belongs to user
    session = await db["chat_sessions"].find_one({"id": session_id, "user_id": current_user.id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get messages for this session
    cur = db["chat_messages"].find({"session_id": session_id}).sort("created_at", 1)
    items = await cur.to_list(100)
    return [ChatMessage(**i) for i in items]

@router.post("/{session_id}/message", response_model=ChatMessage)
async def append_message(session_id: str, req: ChatMessage, current_user=Depends(get_current_active_user)):
    db = get_db()
    
    # Verify session exists and belongs to user
    session = await db["chat_sessions"].find_one({"id": session_id, "user_id": current_user.id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create chat message with proper session_id
    chat_message = ChatMessage(
        role=req.role,
        text=req.text,
        session_id=session_id,
        created_at=datetime.utcnow(),
        sources=req.sources
    )
    
    # Insert chat message to chat_messages collection
    await db["chat_messages"].insert_one(chat_message.dict())
    
    # Update session updated_at timestamp
    await db["chat_sessions"].update_one(
        {"id": session_id}, 
        {"$set": {"updated_at": datetime.utcnow()}}
    )
    
    return chat_message

    



@router.delete("/{session_id}")
async def delete_session(session_id: str, current_user=Depends(get_current_active_user)):
    db = get_db()
    res = await db["chat_sessions"].delete_one({"id": session_id, "user_id": current_user.id})
    await db["chat_messages"].delete_many({"session_id": session_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok"}


# clear all chat messages for a specific session
@router.delete("/{session_id}/messages", response_model=dict)
async def clear_chat_messages(session_id: str, current_user=Depends(get_current_active_user)):
    db = get_db()
    await db["chat_messages"].delete_many({"session_id": session_id})
    return {"status": "ok"}
