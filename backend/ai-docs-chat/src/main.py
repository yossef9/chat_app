# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.database import init_db, close_db
from src.routes.document import router as document_router
from src.routes.user import router as user_router
from src.routes.search_history import router as search_router
from src.routes.chat import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    await init_db()
    yield
    # Close database connection
    close_db()

app = FastAPI(lifespan=lifespan)

# CORS (development-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_router)
app.include_router(user_router)
app.include_router(search_router)
app.include_router(chat_router)

@app.get("/")
async def root():
    return {"message": "AI Docs Chat API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}