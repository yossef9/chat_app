from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
# User Model
class User(BaseModel):
    id: str = Field(..., description="Unique user ID")
    openai_api_key: str = Field(None, description="User's OpenAI API key")
    disabled: bool = Field(None, description="User's OpenAI API key")
    email: str = Field(..., description="User email")
    hashed_password: str = Field(..., description="Hashed password")

class UserOut(BaseModel):
    id: str = Field(..., description="Unique user ID")
    email: str = Field(..., description="User email")



class LoginRequest(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None