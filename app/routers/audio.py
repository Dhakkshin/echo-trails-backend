from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
from datetime import datetime
import json
from typing import Optional
from app.auth.jwt_bearer import JWTBearer
from app.services.audio_service import upload_audio, get_user_audio_files, get_audio_by_id, check_connection
from app.models.audio import AudioModel
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import uuid

router = APIRouter()

def debug_print(request_id: str, message: str, error: Exception = None):
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] [RequestID: {request_id}] {message}")
    if error:
        print(f"[{timestamp}] [RequestID: {request_id}] Error details: {str(error)}")

@router.post("/upload/", dependencies=[Depends(JWTBearer())])
async def upload_audio_file(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    hidden_until: datetime = Form(...),
    current_user: dict = Depends(JWTBearer())
):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"📤 Starting audio upload for user: {current_user['sub']}")
    try:
        # Verify database connection
        debug_print(request_id, "🔌 Verifying database connection")
        try:
            await check_connection()
            debug_print(request_id, "✅ Database connection verified")
        except (ConnectionFailure, ServerSelectionTimeoutError) as ce:
            debug_print(request_id, "❌ Database connection failed", ce)
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please try again later."
            )

        audio_content = await file.read()
        audio_data = AudioModel(
            user_id=current_user["sub"],
            location={"latitude": latitude, "longitude": longitude},
            audio_data=audio_content,
            hidden_until=hidden_until,
            file_name=file.filename
        )
        result = await upload_audio(audio_data)
        debug_print(request_id, f"✅ Audio upload successful - ID: {result['id']}")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Audio upload failed", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/user/files/", dependencies=[Depends(JWTBearer())])
async def list_user_audio_files(current_user: dict = Depends(JWTBearer())):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"📄 Listing audio files for user: {current_user['sub']}")
    try:
        # Verify database connection
        debug_print(request_id, "🔌 Verifying database connection")
        try:
            await check_connection()
            debug_print(request_id, "✅ Database connection verified")
        except (ConnectionFailure, AsyncIOMotorClientSessionError) as ce:
            debug_print(request_id, "❌ Database connection failed", ce)
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please try again later."
            )

        audio_files = await get_user_audio_files(current_user["sub"])
        debug_print(request_id, f"✅ Retrieved {len(audio_files)} audio files")
        return {"audio_files": audio_files}
    except Exception as e:
        debug_print(request_id, "❌ Failed to list audio files", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{audio_id}", dependencies=[Depends(JWTBearer())])
async def get_audio_metadata(audio_id: str, current_user: dict = Depends(JWTBearer())):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"📄 Retrieving metadata for audio ID: {audio_id}")
    try:
        # Verify database connection
        debug_print(request_id, "🔌 Verifying database connection")
        try:
            await check_connection()
            debug_print(request_id, "✅ Database connection verified")
        except (ConnectionFailure, AsyncIOMotorClientSessionError) as ce:
            debug_print(request_id, "❌ Database connection failed", ce)
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please try again later."
            )

        audio = await get_audio_by_id(audio_id, include_data=False)
        if not audio:
            debug_print(request_id, "❌ Audio file not found")
            raise HTTPException(status_code=404, detail="Audio file not found")
        if audio["user_id"] != current_user["sub"]:
            debug_print(request_id, "❌ Not authorized to access this audio file")
            raise HTTPException(status_code=403, detail="Not authorized to access this audio file")
        
        # Convert datetime objects to ISO format strings
        audio["hidden_until"] = audio["hidden_until"].isoformat()
        audio["created_at"] = audio["created_at"].isoformat()
        
        debug_print(request_id, "✅ Metadata retrieval successful")
        return JSONResponse(content=audio)
    except Exception as e:
        debug_print(request_id, "❌ Failed to retrieve audio metadata", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{audio_id}/download", dependencies=[Depends(JWTBearer())])
async def download_audio_file(audio_id: str, current_user: dict = Depends(JWTBearer())):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"📥 Downloading audio file ID: {audio_id}")
    try:
        # Verify database connection
        debug_print(request_id, "🔌 Verifying database connection")
        try:
            await check_connection()
            debug_print(request_id, "✅ Database connection verified")
        except (ConnectionFailure, AsyncIOMotorClientSessionError) as ce:
            debug_print(request_id, "❌ Database connection failed", ce)
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please try again later."
            )

        audio = await get_audio_by_id(audio_id, include_data=True)
        if not audio:
            debug_print(request_id, "❌ Audio file not found")
            raise HTTPException(status_code=404, detail="Audio file not found")
        if audio["user_id"] != current_user["sub"]:
            debug_print(request_id, "❌ Not authorized to access this audio file")
            raise HTTPException(status_code=403, detail="Not authorized to access this audio file")

        def iterfile():
            yield audio["audio_data"]

        debug_print(request_id, "✅ Audio file download successful")
        return StreamingResponse(
            iterfile(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{audio["file_name"]}"',
                "Accept-Ranges": "bytes"
            }
        )
    except Exception as e:
        debug_print(request_id, "❌ Failed to download audio file", e)
        raise HTTPException(status_code=500, detail=str(e))
