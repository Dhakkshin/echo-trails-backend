# app/models/user.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from bson import ObjectId
from datetime import datetime
from pydantic import GetJsonSchemaHandler
from pydantic_core import core_schema

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, values, **kwargs):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: GetJsonSchemaHandler):
        return core_schema.str_schema()

    @classmethod
    def __modify_json_schema__(cls, field_schema, handler):
        field_schema.update(type="string")

class UserCreate(BaseModel):
    username: str = Field(...)
    email: EmailStr = Field(...)
    password: str = Field(...)

class User(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    username: str
    email: EmailStr
    created_at: datetime = Field(default_factory=datetime.utcnow)
    followers: list[str] = Field(default_factory=list)
    following: list[str] = Field(default_factory=list)
    pending_follow_requests: list[str] = Field(default_factory=list)  # stores user IDs of pending requests

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserLogin(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)