from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT = "txt"
    OTHER = "other"


# Document Model
class Document(BaseModel):
    id: str = Field(..., description="Unique document ID")
    type:FileType = Field(FileType.OTHER, description="File Type"),
    user_id: str = Field(..., description="ID of the owner user")
    filename: str = Field(..., description="Original filename")
    gridfs_id: Optional[str] = Field(None, description="ID of file in GridFS")
    status: str = Field("uploaded", description="Document processing status: uploaded/indexing/ready")
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    preview: Optional[str] = Field(None, description="Short text preview of the document content")
