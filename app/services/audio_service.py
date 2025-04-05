from app.models.audio import AudioModel
from app.database.database import get_audio_collection, Database, get_user_collection  
from bson import ObjectId
from fastapi import HTTPException
from datetime import datetime
from app.services.user_service import UserService

async def check_connection():
    """Verify database connection is alive"""
    try:
        return await Database.check_connection()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection error: {str(e)}"
        )

async def validate_recipients(user_id: str, recipient_usernames: list[str]) -> list[str]:
    """Validate recipients and return their user IDs"""
    if not recipient_usernames:
        raise HTTPException(status_code=400, detail="At least one recipient is required")
        
    user_service = UserService()
    user = await user_service.get_user_by_id(user_id)
    following = user.get("following", [])
    
    valid_recipients = []
    for username in recipient_usernames:
        if not username:  # Skip empty usernames
            continue
            
        recipient = await user_service.get_user_by_username(username)
        if not recipient:
            raise HTTPException(status_code=404, detail=f"User {username} not found")
        
        recipient_id = str(recipient["_id"])
        if recipient_id not in following and recipient_id != user_id:
            # Only check following requirement if not self
            raise HTTPException(status_code=403, detail=f"You are not following {username}")
            
        valid_recipients.append(recipient_id)
    
    return valid_recipients

async def upload_audio(audio_data: AudioModel) -> dict:
    try:
        collection = await get_audio_collection()
        user_collection = await get_user_collection()
        audio_dict = audio_data.dict()
        
        # Validate and get recipient IDs (only explicitly specified recipients)
        recipient_ids = await validate_recipients(
            audio_dict["user_id"], 
            audio_dict.get("recipient_usernames", [])
        )
        audio_dict["recipient_ids"] = recipient_ids
        
        # Insert audio
        audio = await collection.insert_one(audio_dict)
        audio_id = str(audio.inserted_id)
        
        # Update only specified recipients' accessible_audio_ids
        for recipient_id in recipient_ids:
            await user_collection.update_one(
                {"_id": ObjectId(recipient_id)},
                {"$addToSet": {"accessible_audio_ids": audio_id}}
            )
            
        return {"id": audio_id}
    except Exception as e:
        print(f"❌ Failed to upload audio: {str(e)}")
        raise

async def get_user_audio_files(user_id: str) -> list:
    collection = await get_audio_collection()
    user_collection = await get_user_collection()
    
    # First get user's accessible_audio_ids
    user = await user_collection.find_one({"_id": ObjectId(user_id)})
    if not user or "accessible_audio_ids" not in user:
        return []
    
    # Convert string IDs to ObjectIds for query
    accessible_ids = [ObjectId(aid) for aid in user.get("accessible_audio_ids", [])]
    
    # Get all audio files the user can access
    cursor = collection.find({"_id": {"$in": accessible_ids}})
    
    audio_files = []
    async for audio in cursor:
        # Get username of audio creator
        creator = await user_collection.find_one({"_id": ObjectId(audio["user_id"])})
        creator_username = creator["username"] if creator else "unknown"
        
        # Format the audio document
        audio_doc = {
            "_id": str(audio["_id"]),
            "user_id": audio["user_id"],
            "username": creator_username,
            "title": audio["title"],
            "file_name": audio["file_name"],
            "location": audio["location"],
            "range": audio["range"],
            "hidden_until": audio["hidden_until"],
            "created_at": audio["created_at"],
            "audio_data": len(audio["audio_data"])
        }
        audio_files.append(audio_doc)
    
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
                "$project": {
                    "_id": 1,
                    "user_id": 1,
                    "title": 1,
                    "file_name": 1,
                    "location": 1,
                    "range": 1,
                    "hidden_until": 1,
                    "created_at": 1,
                    "distance": 1
                }
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

async def delete_audio(audio_id: str, user_id: str) -> bool:
    """Delete an audio file if it belongs to the user"""
    try:
        collection = await get_audio_collection()
        result = await collection.delete_one({
            "_id": ObjectId(audio_id),
            "user_id": user_id  # Ensure user owns the audio
        })
        return result.deleted_count > 0
    except Exception as e:
        print(f"❌ Failed to delete audio: {str(e)}")
        raise

async def get_accessible_audio_files(user_id: str) -> list:
    """Get all audio files that the user can access (own + shared)"""
    try:
        collection = await get_audio_collection()
        cursor = collection.find({
            "$or": [
                {"creator_id": user_id},  # Audio files created by user
                {"recipient_ids": user_id}  # Audio files shared with user
            ]
        })

        audio_files = []
        async for audio in cursor:
            # Get usernames of users it's shared with
            shared_with = []
            if "recipient_ids" in audio:
                user_collection = await get_user_collection()
                for recipient_id in audio["recipient_ids"]:
                    recipient = await user_collection.find_one({"_id": ObjectId(recipient_id)})
                    if recipient:
                        shared_with.append(recipient["username"])

            # Format the response
            audio_files.append({
                "id": str(audio["_id"]),
                "title": audio["title"],
                "location": {
                    "latitude": audio["location"]["coordinates"][1],
                    "longitude": audio["location"]["coordinates"][0]
                },
                "range": audio["range"],
                "hidden_until": audio["hidden_until"].isoformat(),
                "shared_with": shared_with,
                "creator_id": audio["creator_id"]
            })
            
        return audio_files
    except Exception as e:
        print(f"❌ Failed to get accessible audio files: {str(e)}")
        raise