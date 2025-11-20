from fastapi import APIRouter, HTTPException, Depends, status
from fastapi import Body
from fastapi.security import OAuth2PasswordRequestForm
from typing import List
from pymongo import ReturnDocument
import uuid

from src.database import get_db
from src.models.user import UserOut, Token
from src.auth import (
    get_current_active_user,
    get_password_hash,
    authenticate_user,
    create_access_token
)

router = APIRouter(prefix="/users", tags=["users"])


# Register new user
@router.post("/register", response_model=UserOut)
async def register_user(email: str = Body(...), password: str = Body(...)):
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    
    db = get_db()
    existing_user = await db["users"].find_one({"email": email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    hashed_password = get_password_hash(password)
    user_doc = {
        "id": str(uuid.uuid4()),
        "email": email,
        "hashed_password": hashed_password,
        "disabled": False,
    }
    await db["users"].insert_one(user_doc)
    return UserOut(id=user_doc["id"], email=user_doc["email"], disabled=user_doc["disabled"])


# Get all users (safe fields only)
@router.get("/", response_model=List[UserOut])
async def get_all_users():
    db = get_db()
    users_cursor = db["users"].find({}, {"hashed_password": 0})
    users = await users_cursor.to_list(length=100)
    return users


# # Get user by ID
# @router.get("/{user_id}", response_model=UserOut)
# async def get_user_by_id(user_id: str):
#     db = get_db()
#     user = await db["users"].find_one({"id": user_id})
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
#     return user


# # Delete user
# @router.delete("/{user_id}")
# async def delete_user(user_id: str):
#     db = get_db()
#     result = await db["users"].delete_one({"id": user_id})
#     if result.deleted_count == 0:
#         raise HTTPException(status_code=404, detail="User not found")
#     return {"detail": "User deleted successfully"}


# Login and get token
@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserOut)
async def read_current_user(current_user=Depends(get_current_active_user)):
    return UserOut(id=current_user.id, email=current_user.email)

@router.post("/save-api-key", response_model=UserOut)
async def save_api_key(
    openai_api_key:str,
    current_user=Depends(get_current_active_user)
):
    if not openai_api_key:
        raise HTTPException(status_code=400, detail="openai_api_key is required")

    db = get_db()
    updated_user = await db['users'].find_one_and_update(
        {"id": current_user.id},
        {"$set": {"openai_api_key": openai_api_key}},
        return_document=ReturnDocument.AFTER
    )

    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserOut(id=updated_user["id"], email=updated_user["email"])


@router.delete("/api-key", response_model=UserOut)
async def delete_api_key(current_user=Depends(get_current_active_user)):
    db = get_db()
    updated_user = await db['users'].find_one_and_update(
        {"id": current_user.id},
        {"$unset": {"openai_api_key": ""}},
        return_document=ReturnDocument.AFTER
    )

    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserOut(id=updated_user["id"], email=updated_user["email"])
    
