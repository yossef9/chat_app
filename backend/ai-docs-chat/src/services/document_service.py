# src/services/document_service.py
import uuid
from datetime import datetime
from typing import Optional
import io
import PyPDF2
import docx
from bs4 import BeautifulSoup

from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from fastapi import UploadFile
from src.models.document import Document, FileType
from src.database import get_db
from langchain.schema import Document as LangchainDocument
from typing import List, Dict, Any, Optional


class DocumentService:
    async def save_file_to_gridfs(file: UploadFile, user_id: str) -> str:
        db = get_db()
        fs_bucket = AsyncIOMotorGridFSBucket(db)

        content = await file.read()
        gridfs_id = await fs_bucket.upload_from_stream(
            file.filename,
            content,
            metadata={"user_id": user_id}
        )
        return str(gridfs_id)

    
    async def extract_text(self, filename: str, file_bytes: bytes) -> str:
        filetype = self.get_filetype_from_filename(filename)
        
        try:
            if filetype == FileType.PDF:
                return await self._extract_pdf(file_bytes)
            elif filetype==FileType.DOCX:
                return await self._extract_docx(file_bytes)
            elif filetype==FileType.HTML:
                return await self._extract_html(file_bytes)
            elif filetype == FileType.TXT:
                return file_bytes.decode('utf-8')
            else:
                # Try to decode as text
                try:
                    return file_bytes.decode('utf-8')
                except:
                    raise ValueError(f"Unsupported file format: {filetype}")
        except Exception as e:
            raise Exception(f"Failed to extract text from {filename}: {str(e)}")
    
    async def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF"""
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    async def _extract_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX"""
        doc_file = io.BytesIO(file_bytes)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    async def _extract_html(self, file_bytes: bytes) -> str:
        """Extract text from HTML"""
        soup = BeautifulSoup(file_bytes, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()
    
    def _split_text(self, text: str) -> List[LangchainDocument]:
        """Split text into chunks"""
        if not text.strip():
            raise ValueError("Empty text content")
        
        # Use LangChain's document splitter
        documents = self.text_splitter.create_documents([text])
        return documents
    
    @staticmethod
    def get_filetype_from_filename(filename: str) -> FileType:
        ext = filename.lower()
        if ext.endswith(".pdf"):
            return FileType.PDF
        elif ext.endswith(".docx") or ext.endswith(".doc"):
            return FileType.DOCX
        elif ext.endswith(".html") or ext.endswith(".htm"):
            return FileType.HTML
        elif ext.endswith(".txt"):
            return FileType.TXT
        else:
            return FileType.OTHER


document_service = DocumentService()








