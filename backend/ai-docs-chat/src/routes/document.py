# src/documents.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import io
from src.models.document import Document,FileType
from src.database import get_db
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from fastapi.responses import StreamingResponse
from bson import ObjectId
from typing import List
from fastapi import Query
import uuid
from src.auth import get_current_active_user  # your JWT auth dependency
from src.models.user import User
from fastapi import BackgroundTasks
from src.services.document_service import document_service
from src.services.rag_service import rag_service
router = APIRouter(prefix="/documents", tags=["documents"])


from src.models.document import FileType




# Create document
@router.post("/", response_model=Document)
async def create_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    doc_id = str(uuid.uuid4())
    fs_bucket = AsyncIOMotorGridFSBucket(db)

    # Upload file to GridFS
    file_content = await file.read()
    gridfs_id = await fs_bucket.upload_from_stream(
        file.filename,
        file_content,
        metadata={"user_id": current_user.id, "document_id": doc_id}
    )

    document = Document(
        id=doc_id,
        user_id=current_user.id,
        type=document_service.get_filetype_from_filename(file.filename),
        filename=file.filename,
        gridfs_id=str(gridfs_id),
        status="uploaded"
    )

    await db.documents.insert_one(document.dict())

    # FIX: Use file_content instead of file_bytes
    background_tasks.add_task(rag_service.add_document, document, file_content)
    return document


# Get all documents for the current user
@router.get("/me", response_model=List[Document])
async def get_user_documents(status: str | None = Query(None), current_user: User = Depends(get_current_active_user)):
    db = get_db()
    query = {"user_id": current_user.id}
    if status:
        query["status"] = status
    documents = await db.documents.find(query).to_list(100)
    return [Document(**doc) for doc in documents]




# # Download document content
# @router.get("/{doc_id}/download")
# async def download_document(
#     doc_id: str,
#     current_user: User = Depends(get_current_active_user)
# ):
#     db = get_db()
#     fs_bucket = AsyncIOMotorGridFSBucket(db)

#     document = await db.documents.find_one({"id": doc_id, "user_id": current_user.id})
#     if not document:
#         raise HTTPException(status_code=404, detail="Document not found")

#     gridfs_id = document.get("gridfs_id")
#     if not gridfs_id:
#         raise HTTPException(status_code=404, detail="Document content not found")

#     try:
#         stream = io.BytesIO()
#         await fs_bucket.download_to_stream(ObjectId(gridfs_id), stream)
#         stream.seek(0)
#         return StreamingResponse(
#             stream,
#             media_type="application/octet-stream",
#             headers={"Content-Disposition": f'attachment; filename="{document["filename"]}"'}
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


# Delete document
@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_active_user)
):
    db = get_db()
    document = await db.documents.find_one({"id": doc_id, "user_id": current_user.id})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        fs_bucket = AsyncIOMotorGridFSBucket(db)
        gridfs_id = document.get("gridfs_id")
        if gridfs_id:
            await fs_bucket.delete(ObjectId(gridfs_id))
        await db.documents.delete_one({"id": doc_id, "user_id": current_user.id})
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
