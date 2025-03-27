# app/services/user_service.py
from app.models.user import User, UserCreate
from app.database.database import get_user_collection, Database
from app.auth.jwt_handler import hash_password
from bson import ObjectId
from fastapi import HTTPException, status
from datetime import datetime

class UserService:
    async def get_collection(self):
        return await get_user_collection()

    async def create_user(self, user: UserCreate):
        collection = await self.get_collection()
        existing_user = await collection.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = hash_password(user.password)
        user_dict = user.model_dump(by_alias=True)
        user_dict["password"] = hashed_password
        user_dict["created_at"] = datetime.utcnow()
        
        new_user = await collection.insert_one(user_dict)
        created_user = await collection.find_one({"_id": new_user.inserted_id})
        
        # Create response without password
        response_data = {
            "_id": str(created_user["_id"]),
            "username": created_user["username"],
            "email": created_user["email"],
            "created_at": created_user["created_at"]
        }
        
        return User(**response_data)

    async def get_user_by_email(self, email: str):
        collection = await self.get_collection()
        return await collection.find_one({"email": email})

    async def get_user_by_id(self, user_id: str):
        collection = await self.get_collection()
        user = await collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user["_id"] = str(user["_id"])  # Convert ObjectId to string
        return user
    
    async def hello_world(self):
        return {"message": "Hello, this is from user servives!"}