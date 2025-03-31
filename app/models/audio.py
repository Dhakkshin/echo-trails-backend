from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId

class AudioModel(BaseModel):
    user_id: str
    title: str = Field(..., description="Title of the audio recording")
    location: dict = Field(
        ..., 
        example={
            "type": "Point",
            "coordinates": [0.0, 0.0]  # [longitude, latitude]
        }
    )
    audio_data: bytes
    hidden_until: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_name: str
    range: float = Field(..., description="Distance range in meters")
