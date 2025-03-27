from app.models.audio import AudioModel
from app.database.database import get_audio_collection, Database
from bson import ObjectId
from fastapi import HTTPException
from datetime import datetime

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

async def get_nearby_audio_files(latitude: float, longitude: float, user_id: str):
    """
    Uses MongoDB's $geoWithin operator to find audio files within range
    """
    try:
        collection = await get_audio_collection()
        
        pipeline = [
            {
                "$geoNear": {
                    "near": {
                        "type": "Point",
                        "coordinates": [longitude, latitude]  # MongoDB uses [long, lat]
                    },
                    "distanceField": "calcDistance",  # Change to a temporary field name
                    "spherical": True,
                    "query": {
                        "hidden_until": {"$lte": datetime.utcnow()},  # Only show unhidden files
                        "user_id": user_id  # Add user_id filter
                    }
                }
            },
            {
                "$match": {
                    "$expr": {
                        "$lte": ["$calcDistance", "$range"]  # Check if distance is within range
                    }
                }
            },
            {
                "$addFields": {
                    "distance": "$calcDistance",  # Add distance as a new field
                    "location": {
                        "latitude": {"$arrayElemAt": ["$location.coordinates", 1]},
                        "longitude": {"$arrayElemAt": ["$location.coordinates", 0]}
                    }
                }
            },
            {
                "$unset": ["calcDistance", "audio_data"]  # Remove temporary field and audio data
            }
        ]
        
        nearby_files = await collection.aggregate(pipeline).to_list(length=None)
        
        # Convert ObjectId and datetime to strings for JSON serialization
        for file in nearby_files:
            file["_id"] = str(file["_id"])
            file["hidden_until"] = file["hidden_until"].isoformat()
            file["created_at"] = file["created_at"].isoformat()
            file["distance"] = round(file["distance"], 2)  # Round to 2 decimal places
            
        return nearby_files
        
    except Exception as e:
        print(f"Error finding nearby audio files: {str(e)}")
        raise