from app.database.database import audio_collection
from app.models.audio import AudioModel
from bson import ObjectId
# import logging

# logger = logging.getLogger(__name__)

async def upload_audio(audio_data: AudioModel) -> dict:
    try:
        # logger.info("Converting AudioModel to dict...")
        audio_dict = audio_data.dict()
        # logger.debug(f"Audio metadata: user_id={audio_dict['user_id']}, filename={audio_dict['file_name']}")
        
        # logger.info("Inserting audio into database...")
        audio = await audio_collection.insert_one(audio_dict)
        # logger.info(f"Audio inserted successfully with ID: {audio.inserted_id}")
        
        return {"id": str(audio.inserted_id)}
    except Exception as e:
        # logger.error(f"Failed to upload audio to database: {str(e)}", exc_info=True)
        raise

async def get_user_audio_files(user_id: str) -> list:
    cursor = audio_collection.find({"user_id": user_id})
    audio_files = []
    async for audio in cursor:
        audio["_id"] = str(audio["_id"])
        audio["audio_data"] = len(audio["audio_data"])  # Return only the size of audio data
        audio_files.append(audio)
    return audio_files

async def get_audio_by_id(audio_id: str, include_data: bool = True) -> dict:
    projection = None if include_data else {"audio_data": 0}
    audio = await audio_collection.find_one({"_id": ObjectId(audio_id)}, projection)
    if audio:
        audio["_id"] = str(audio["_id"])
        return audio
    return None
