from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class Chat(BaseModel):
    id: str = Field(..., description="Unique identifier for the chat message")
    openai_api_key: str = Field(..., description="OpenAI API key used for the chat")
    user_id: str = Field(..., description="Associated user ID")
    search_message: str = Field(..., description="The message or query sent by the user")
    response: str = Field(..., description="The response from the AI")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last updated timestamp")

class ChatOut(BaseModel):
    id: str
    user_id: str
    search_message: str
    response: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class ChatList(BaseModel):
    chats: list[ChatOut]
    total: int
