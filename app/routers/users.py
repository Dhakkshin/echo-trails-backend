# app/routers/users.py
from fastapi import APIRouter, HTTPException, Depends, status
from app.models.user import User, UserLogin, UserCreate
from app.auth.jwt_handler import create_access_token, verify_password
from app.services.user_service import UserService
from app.auth.jwt_bearer import JWTBearer
from app.database.database import Database
from datetime import datetime
import uuid
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

router = APIRouter()

def debug_print(request_id: str, message: str, error: Exception = None):
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] [RequestID: {request_id}] {message}")
    if error:
        print(f"[{timestamp}] [RequestID: {request_id}] Error details: {str(error)}")

@router.post("/register/", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, user_service: UserService = Depends()):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"📝 Starting registration process for user: {user.email}")
    try:
        created_user = await user_service.create_user(user)
        debug_print(request_id, f"✅ User registration successful - Username: {created_user.username}, Email: {created_user.email}")
        return created_user
    except HTTPException as he:
        debug_print(request_id, f"❌ Registration failed - HTTP Error", he)
        raise
    except Exception as e:
        debug_print(request_id, f"⚠️ Unexpected registration error", e)
        raise

@router.post("/login/")
async def login_user(user: UserLogin, user_service: UserService = Depends()):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"🔐 Login attempt initiated - Email: {user.email}")
    try:
        # Check database connection
        debug_print(request_id, "🔌 Verifying database connection")
        try:
            await Database.check_connection()
            debug_print(request_id, "✅ Database connection verified")
        except (ConnectionFailure, ServerSelectionTimeoutError) as ce:
            debug_print(request_id, "❌ Database connection failed", ce)
            raise HTTPException(
                status_code=503,
                detail="Database connection error. Please try again later."
            )

        db_user = await user_service.get_user_by_email(user.email)
        debug_print(request_id, f"🔍 User lookup result - Found: {db_user is not None}")
        
        if not db_user:
            debug_print(request_id, "❌ Authentication failed - User not found")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        is_valid = verify_password(user.password, db_user["password"])
        debug_print(request_id, f"🔑 Password verification - Valid: {is_valid}")
        
        if not is_valid:
            debug_print(request_id, "❌ Authentication failed - Invalid password")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        access_token = create_access_token(data={"sub": str(db_user["_id"])})
        debug_print(request_id, f"✅ Login successful - Token generated for user: {db_user['email']}")
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "username": db_user["username"]
        }
    except HTTPException as he:
        debug_print(request_id, "❌ Login process failed - HTTP Exception", he)
        raise
    except Exception as e:
        debug_print(request_id, "❌ Login process failed - Unexpected error", e)
        # Return a more specific error message
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Internal server error during login",
                "error_type": e.__class__.__name__,
                "error": str(e)
            }
        )

@router.get("/hello/")
async def hello_world(user_service: UserService = Depends()):
    request_id = str(uuid.uuid4())
    debug_print(request_id, "🌍 Hello world endpoint called")
    try:
        result = await user_service.hello_world()
        debug_print(request_id, "✅ Hello world response generated successfully")
        return result
    except Exception as e:
        debug_print(request_id, f"⚠️ Error in hello world endpoint", e)
        raise

@router.get("/identify", dependencies=[Depends(JWTBearer())])
async def identify_user(token: str = Depends(JWTBearer()), user_service: UserService = Depends()):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"🔍 User identification attempt - Token Subject: {token.get('sub')}")
    try:
        user_id = token.get("sub")
        debug_print(request_id, f"👤 Fetching user profile - ID: {user_id}")
        
        user_data = await user_service.get_user_by_id(user_id)
        debug_print(request_id, f"📋 User data retrieval - Success: {user_data is not None}")
        
        if not user_data:
            debug_print(request_id, f"❌ User not found in database - ID: {user_id}")
            raise HTTPException(status_code=404, detail="User not found in database")

        response = {
            "status": "success",
            "token_info": {
                "user_id": user_id,
                "issued_at": datetime.utcfromtimestamp(token.get("iat")).isoformat(),
                "expires_at": datetime.utcfromtimestamp(token.get("exp")).isoformat(),
                "token_valid": True,
                "scopes": token.get("scopes", [])
            },
            "user_data": {
                "id": user_data["_id"],
                "username": user_data.get("username"),
                "email": user_data.get("email"),
                "created_at": user_data.get("created_at"),
            }
        }
        debug_print(request_id, f"✅ User identification successful - Username: {user_data.get('username')}")
        return response

    except HTTPException as he:
        debug_print(request_id, "❌ User identification failed - HTTP Exception", he)
        raise
    except Exception as e:
        debug_print(request_id, "⚠️ User identification failed - Unexpected error", e)
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "Invalid token or user data",
                "error": str(e)
            }
        )

# @router.post("/follow/{username}")
# async def follow_user(
#     username: str,
#     token: dict = Depends(JWTBearer()),
#     user_service: UserService = Depends()
# ):
#     request_id = str(uuid.uuid4())
#     debug_print(request_id, f"👥 Follow request initiated - Target username: {username}")
#     try:
#         follower_id = token.get("sub")
#         result = await user_service.follow_user(follower_id, username)
#         debug_print(request_id, f"✅ Follow successful - User {follower_id} followed {username}")
#         return result
#     except Exception as e:
#         debug_print(request_id, "❌ Follow request failed", e)
#         raise

@router.post("/unfollow/{username}")
async def unfollow_user(
    username: str,
    token: dict = Depends(JWTBearer()),
    user_service: UserService = Depends()
):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"👥 Unfollow request initiated - Target username: {username}")
    try:
        follower_id = token.get("sub")
        result = await user_service.stop_following(follower_id, username)  # Changed from unfollow_user to stop_following
        debug_print(request_id, f"✅ Unfollow successful - User {follower_id} unfollowed {username}")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Unfollow request failed", e)
        raise

@router.post("/follow/request/{username}")
async def send_follow_request(
    username: str,
    token: dict = Depends(JWTBearer()),
    user_service: UserService = Depends()
):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"👥 Follow request initiated - Target username: {username}")
    try:
        sender_id = token.get("sub")
        result = await user_service.send_follow_request(sender_id, username)
        debug_print(request_id, f"✅ Follow request sent - From {sender_id} to {username}")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Follow request failed", e)
        raise

@router.post("/follow/accept/{requester_id}")
async def accept_follow_request(
    requester_id: str,
    token: dict = Depends(JWTBearer()),
    user_service: UserService = Depends()
):
    request_id = str(uuid.uuid4())
    try:
        user_id = token.get("sub")
        result = await user_service.accept_follow_request(user_id, requester_id)
        debug_print(request_id, f"✅ Follow request accepted - From {requester_id}")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Accept follow request failed", e)
        raise

@router.post("/follow/reject/{requester_id}")
async def reject_follow_request(
    requester_id: str,
    token: dict = Depends(JWTBearer()),
    user_service: UserService = Depends()
):
    request_id = str(uuid.uuid4())
    try:
        user_id = token.get("sub")
        result = await user_service.reject_follow_request(user_id, requester_id)
        debug_print(request_id, f"✅ Follow request rejected - From {requester_id}")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Reject follow request failed", e)
        raise

@router.get("/follow/requests/pending")
async def get_pending_follow_requests(
    token: dict = Depends(JWTBearer()),
    user_service: UserService = Depends()
):
    request_id = str(uuid.uuid4())
    try:
        user_id = token.get("sub")
        result = await user_service.get_pending_follow_requests(user_id)
        debug_print(request_id, f"✅ Retrieved pending follow requests")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Get pending requests failed", e)
        raise

@router.post("/unfollow/{username}")
async def stop_following(
    username: str,
    token: dict = Depends(JWTBearer()),
    user_service: UserService = Depends()
):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"👥 Stop following request initiated - Target username: {username}")
    try:
        follower_id = token.get("sub")
        result = await user_service.stop_following(follower_id, username)
        debug_print(request_id, f"✅ Unfollowed successful - User {follower_id} stopped following {username}")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Unfollow request failed", e)
        raise

@router.post("/followers/remove/{username}")
async def remove_follower(
    username: str,
    token: dict = Depends(JWTBearer()),
    user_service: UserService = Depends()
):
    request_id = str(uuid.uuid4())
    debug_print(request_id, f"👥 Remove follower request initiated - Follower username: {username}")
    try:
        user_id = token.get("sub")
        result = await user_service.remove_follower(user_id, username)
        debug_print(request_id, f"✅ Remove follower successful - Removed {username} from followers")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Remove follower request failed", e)
        raise

@router.get("/following")
async def get_following_users(
    token: dict = Depends(JWTBearer()),
    user_service: UserService = Depends()
):
    request_id = str(uuid.uuid4())
    try:
        user_id = token.get("sub")
        result = await user_service.get_following_users(user_id)
        debug_print(request_id, f"✅ Retrieved following users list for user {user_id}")
        return result
    except Exception as e:
        debug_print(request_id, "❌ Get following users failed", e)
        raise
