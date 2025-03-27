from app.models.audio import AudioModel
from app.database.database import get_audio_collection, Database
from bson import ObjectId
from fastapi import HTTPException

async def check_connection():
    """Verify database connection is alive"""
    try:
        return await Database.check_connection()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {str(e)}"
        )

async def upload_audio(audio_data: AudioModel) -> dict:
    try:
        collection = await get_audio_collection()
        audio_dict = audio_data.dict()
        audio = await collection.insert_one(audio_dict)
        return {"id": str(audio.inserted_id)}
    except Exception as e:
        print(f"❌ Failed to upload audio: {str(e)}")
        raise

async def get_user_audio_files(user_id: str) -> list:
    collection = await get_audio_collection()
    cursor = collection.find({"user_id": user_id})
    audio_files = []
    async for audio in cursor:
        audio["_id"] = str(audio["_id"])
        audio["audio_data"] = len(audio["audio_data"])
        audio_files.append(audio)
    return audio_files

async def get_audio_by_id(audio_id: str, include_data: bool = True) -> dict:
    collection = await get_audio_collection()
    projection = None if include_data else {"audio_data": 0}
    audio = await collection.find_one({"_id": ObjectId(audio_id)}, projection)
    if audio:
        audio["_id"] = str(audio["_id"])
    return audio
