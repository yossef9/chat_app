from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant' | 'system'
    text: str
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[List[dict]] = None

class CreateSessionRequest(BaseModel):
    name: Optional[str] = None
    document_ids: Optional[List[str]] = None


class ChatSession(BaseModel):
    id: str
    user_id: str
    name: Optional[str] = None
    document_ids: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)



class SearchHistory(BaseModel):
    id: str = Field(..., description="Unique identifier for the search history")
    openai_api_key: str = Field(..., description="OpenAI API key used for the search")
    user_id: str = Field(..., description="Associated user ID")
    search_message: str = Field(..., description="The message or query sent by the user")
    response: str = Field(..., description="The response from the user")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last updated timestamp")
