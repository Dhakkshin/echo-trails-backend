from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId

class AudioModel(BaseModel):
    user_id: str
    location: dict = Field(..., example={"latitude": 0.0, "longitude": 0.0})
    audio_data: bytes
    hidden_until: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_name: str
