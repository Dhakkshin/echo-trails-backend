# app/routers/users.py
from fastapi import APIRouter, HTTPException, Depends, status
from app.models.user import User, UserLogin, UserCreate
from app.auth.jwt_handler import create_access_token, verify_password
from app.services.user_service import UserService
from app.auth.jwt_bearer import JWTBearer
from datetime import datetime
import uuid

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
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        debug_print(request_id, "❌ Login process failed", e)
        raise

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
