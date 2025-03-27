from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
from datetime import datetime
import json
from typing import Optional
from app.auth.jwt_bearer import JWTBearer
from app.services.audio_service import upload_audio, get_user_audio_files, get_audio_by_id
from app.models.audio import AudioModel
# import logging

# Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload/", dependencies=[Depends(JWTBearer())])
async def upload_audio_file(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    hidden_until: datetime = Form(...),
    current_user: dict = Depends(JWTBearer())
):
    try:
        # logger.debug(f"Starting audio upload process for user: {current_user}")
        # logger.debug(f"File details - Filename: {file.filename}, Content-Type: {file.content_type}")
        audio_content = await file.read()
        audio_data = AudioModel(
            user_id=current_user["sub"],
            location={"latitude": latitude, "longitude": longitude},
            audio_data=audio_content,
            hidden_until=hidden_until,
            file_name=file.filename
        )
        result = await upload_audio(audio_data)
        return result
    except Exception as e:
        # logger.error(f"Error during audio upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/user/files/", dependencies=[Depends(JWTBearer())])
async def list_user_audio_files(current_user: dict = Depends(JWTBearer())):
    try:
        audio_files = await get_user_audio_files(current_user["sub"])
        return {"audio_files": audio_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{audio_id}", dependencies=[Depends(JWTBearer())])
async def get_audio_metadata(audio_id: str, current_user: dict = Depends(JWTBearer())):
    try:
        audio = await get_audio_by_id(audio_id, include_data=False)
        if not audio:
            raise HTTPException(status_code=404, detail="Audio file not found")
        if audio["user_id"] != current_user["sub"]:
            raise HTTPException(status_code=403, detail="Not authorized to access this audio file")
        
        # Convert datetime objects to ISO format strings
        audio["hidden_until"] = audio["hidden_until"].isoformat()
        audio["created_at"] = audio["created_at"].isoformat()
        
        return JSONResponse(content=audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{audio_id}/download", dependencies=[Depends(JWTBearer())])
async def download_audio_file(audio_id: str, current_user: dict = Depends(JWTBearer())):
    try:
        audio = await get_audio_by_id(audio_id, include_data=True)
        if not audio:
            raise HTTPException(status_code=404, detail="Audio file not found")
        if audio["user_id"] != current_user["sub"]:
            raise HTTPException(status_code=403, detail="Not authorized to access this audio file")

        def iterfile():
            yield audio["audio_data"]

        return StreamingResponse(
            iterfile(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="{audio["file_name"]}"',
                "Accept-Ranges": "bytes"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
