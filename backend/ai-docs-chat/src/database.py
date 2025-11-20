# src/database.py
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_db = None
mongodb_client = None

async def init_db():
    global mongo_db, mongodb_client
    if mongo_db is None:
        try:
            mongodb_client =  AsyncIOMotorClient(
                MONGO_URI,
                tls=True,
                tlsAllowInvalidCertificates=True,
                retryWrites=True,
                serverSelectionTimeoutMS=10000
            )
            # Test connection
            await mongodb_client.admin.command('ping')
            mongo_db =  mongodb_client["ai-docs-chat"]
            print("MongoDB connected successfully.")
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            raise

def close_db():
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        print("MongoDB disconnected.")

def get_db():
    if mongo_db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return mongo_db
    