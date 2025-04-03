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
        
        # Check for existing email
        existing_email = await collection.find_one({"email": user.email})
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Check for existing username
        existing_username = await collection.find_one({"username": user.username})
        if existing_username:
            raise HTTPException(status_code=400, detail="Username already taken")

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

    async def follow_user(self, follower_id: str, username_to_follow: str):
        collection = await self.get_collection()
        
        # Get user to follow
        user_to_follow = await collection.find_one({"username": username_to_follow})
        if not user_to_follow:
            raise HTTPException(status_code=404, detail="User not found")
            
        target_id = str(user_to_follow["_id"])
        
        # Prevent self-following
        if follower_id == target_id:
            raise HTTPException(status_code=400, detail="Cannot follow yourself")
            
        # Check if already following
        follower = await collection.find_one({"_id": ObjectId(follower_id)})
        if target_id in follower.get("following", []):
            raise HTTPException(status_code=400, detail="Already following this user")
            
        # Update both users' followers/following lists
        await collection.update_one(
            {"_id": ObjectId(follower_id)},
            {"$addToSet": {"following": target_id}}
        )
        
        await collection.update_one(
            {"_id": ObjectId(target_id)},
            {"$addToSet": {"followers": follower_id}}
        )
        
        return {"message": f"Successfully followed {username_to_follow}"}

    async def send_follow_request(self, sender_id: str, username_to_follow: str):
        collection = await self.get_collection()
        
        # Get user to follow
        user_to_follow = await collection.find_one({"username": username_to_follow})
        if not user_to_follow:
            raise HTTPException(status_code=404, detail="User not found")
            
        target_id = str(user_to_follow["_id"])
        
        # Prevent self-following
        if sender_id == target_id:
            raise HTTPException(status_code=400, detail="Cannot follow yourself")
            
        # Check if already following
        sender = await collection.find_one({"_id": ObjectId(sender_id)})
        if target_id in sender.get("following", []):
            raise HTTPException(status_code=400, detail="Already following this user")
            
        # Check if request already pending
        if sender_id in user_to_follow.get("pending_follow_requests", []):
            raise HTTPException(status_code=400, detail="Follow request already pending")
            
        # Add to pending requests
        await collection.update_one(
            {"_id": ObjectId(target_id)},
            {"$addToSet": {"pending_follow_requests": sender_id}}
        )
        
        return {"message": f"Follow request sent to {username_to_follow}"}

    async def accept_follow_request(self, user_id: str, requester_id: str):
        collection = await self.get_collection()
        
        # Verify request exists
        user = await collection.find_one({"_id": ObjectId(user_id)})
        if requester_id not in user.get("pending_follow_requests", []):
            raise HTTPException(status_code=404, detail="No pending follow request found")
        
        # Remove from pending and add to followers/following
        await collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$pull": {"pending_follow_requests": requester_id},
                "$addToSet": {"followers": requester_id}
            }
        )
        
        await collection.update_one(
            {"_id": ObjectId(requester_id)},
            {"$addToSet": {"following": user_id}}
        )
        
        return {"message": "Follow request accepted"}

    async def reject_follow_request(self, user_id: str, requester_id: str):
        collection = await self.get_collection()
        
        # Remove from pending requests
        await collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$pull": {"pending_follow_requests": requester_id}}
        )
        
        return {"message": "Follow request rejected"}

    async def get_pending_follow_requests(self, user_id: str):
        collection = await self.get_collection()
        user = await collection.find_one({"_id": ObjectId(user_id)})
        
        # Get details of users who sent requests
        pending_requests = user.get("pending_follow_requests", [])
        requesters = []
        
        for requester_id in pending_requests:
            requester = await collection.find_one({"_id": ObjectId(requester_id)})
            if requester:
                requesters.append({
                    "id": str(requester["_id"]),
                    "username": requester["username"]
                })
                
        return requesters

    async def stop_following(self, follower_id: str, username_to_unfollow: str):
        collection = await self.get_collection()
        
        # Get user to unfollow
        user_to_unfollow = await collection.find_one({"username": username_to_unfollow})
        if not user_to_unfollow:
            raise HTTPException(status_code=404, detail="User not found")
            
        target_id = str(user_to_unfollow["_id"])
        
        # Check if actually following
        follower = await collection.find_one({"_id": ObjectId(follower_id)})
        if target_id not in follower.get("following", []):
            raise HTTPException(status_code=400, detail="Not following this user")
        
        # Remove from following/followers lists
        await collection.update_one(
            {"_id": ObjectId(follower_id)},
            {"$pull": {"following": target_id}}
        )
        
        await collection.update_one(
            {"_id": ObjectId(target_id)},
            {"$pull": {"followers": follower_id}}
        )
        
        return {"message": f"Successfully stopped following {username_to_unfollow}"}

    async def remove_follower(self, user_id: str, follower_username: str):
        collection = await self.get_collection()
        
        # Get follower to remove
        follower = await collection.find_one({"username": follower_username})
        if not follower:
            raise HTTPException(status_code=404, detail="User not found")
            
        follower_id = str(follower["_id"])
        
        # Check if actually a follower
        user = await collection.find_one({"_id": ObjectId(user_id)})
        if follower_id not in user.get("followers", []):
            raise HTTPException(status_code=400, detail="This user is not your follower")
        
        # Remove from following/followers lists
        await collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$pull": {"followers": follower_id}}
        )
        
        await collection.update_one(
            {"_id": ObjectId(follower_id)},
            {"$pull": {"following": user_id}}
        )
        
        return {"message": f"Successfully removed {follower_username} from your followers"}

    async def get_following_users(self, user_id: str):
        collection = await self.get_collection()
        user = await collection.find_one({"_id": ObjectId(user_id)})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        following_list = user.get("following", [])
        following_users = []
        
        for following_id in following_list:
            following_user = await collection.find_one({"_id": ObjectId(following_id)})
            if following_user:
                following_users.append({
                    "id": str(following_user["_id"]),
                    "username": following_user["username"]
                })
                
        return following_users