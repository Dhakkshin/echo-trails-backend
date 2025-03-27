# app/database/database.py
import motor.motor_asyncio
from app.config import MONGO_DETAILS
import asyncio
from typing import Optional
from datetime import datetime, timedelta

class Database:
    client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
    last_connection_attempt: datetime = None
    connection_retry_delay: int = 5  # seconds
    max_retries: int = 3
    
    @classmethod
    async def connect_db(cls, retry_count: int = 0):
        if cls.client is not None:
            return

        # Check if we need to wait before retrying
        if cls.last_connection_attempt:
            time_since_last_attempt = (datetime.utcnow() - cls.last_connection_attempt).total_seconds()
            if time_since_last_attempt < cls.connection_retry_delay:
                await asyncio.sleep(cls.connection_retry_delay - time_since_last_attempt)

        cls.last_connection_attempt = datetime.utcnow()
        
        try:
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(
                MONGO_DETAILS,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000,
                retryWrites=True,
                retryReads=True
            )
            # Verify connection
            await cls.client.admin.command('ping')
            print(f"✅ Connected to MongoDB (Attempt {retry_count + 1})")
        except Exception as e:
            print(f"❌ MongoDB connection error (Attempt {retry_count + 1}): {str(e)}")
            cls.client = None
            
            if retry_count < cls.max_retries:
                print(f"🔄 Retrying connection in {cls.connection_retry_delay} seconds...")
                await asyncio.sleep(cls.connection_retry_delay)
                return await cls.connect_db(retry_count + 1)
            raise

    @classmethod
    async def check_connection(cls) -> bool:
        """Check if connection is alive and reconnect if needed"""
        if cls.client is None:
            await cls.connect_db()
            return True

        try:
            await cls.client.admin.command('ping')
            return True
        except:
            cls.client = None
            await cls.connect_db()
            return True

    @classmethod
    async def close_db(cls):
        if cls.client is not None:
            cls.client.close()
            cls.client = None
            cls.last_connection_attempt = None
            print("📤 Closed MongoDB connection")

    @classmethod
    async def get_db(cls):
        await cls.check_connection()
        return cls.client

# Create database instance
db = Database()

# Get database and collections
async def get_database():
    client = await db.get_db()
    return client.audio_app_db

async def get_user_collection():
    database = await get_database()
    return database.users_collection

async def get_audio_collection():
    database = await get_database()
    return database.audio_collection