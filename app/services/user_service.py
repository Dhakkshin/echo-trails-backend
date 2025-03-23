# app/services/user_service.py
from app.models.user import User, UserCreate
from app.database.database import user_collection
from app.auth.auth_utils import hash_password
from bson import ObjectId
from fastapi import HTTPException, status

class UserService:
    async def create_user(self, user: UserCreate):
        existing_user = await user_collection.find_one({"email": user.email})
        print(f"Existing user: {existing_user}")
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = hash_password(user.password)
        # user_dict = user.dict(by_alias=True)
        user_dict = user.model_dump(by_alias=True)
        user_dict["password"] = hashed_password
        new_user = await user_collection.insert_one(user_dict)
        created_user = await user_collection.find_one({"_id": new_user.inserted_id})
        created_user["_id"] = str(created_user["_id"])
        created_user.pop("password") # remove the password from the response.
        return User(**created_user)

    async def get_user_by_email(self, email: str):
        return await user_collection.find_one({"email": email})

    async def get_user_by_id(self, user_id: str):
        return await user_collection.find_one({"_id": ObjectId(user_id)})
    
    async def hello_world(self):
        return {"message": "Hello, this is from user servives!"}