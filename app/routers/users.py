# app/routers/users.py
from fastapi import APIRouter, HTTPException, Depends, status
from app.models.user import User, UserLogin, UserCreate
from app.auth.jwt_handler import create_access_token, verify_password
from app.services.user_service import UserService
from app.auth.jwt_bearer import JWTBearer
from datetime import datetime

router = APIRouter()

@router.post("/register/", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, user_service: UserService = Depends()):
    return await user_service.create_user(user)

@router.post("/login/")
async def login_user(user: UserLogin, user_service: UserService = Depends()):
    print(f"Login attempt for email: {user.email}")
    try:
        db_user = await user_service.get_user_by_email(user.email)
        print(f"Database user found: {db_user is not None}")
        
        if not db_user:
            print("No user found with this email")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        is_valid = verify_password(user.password, db_user["password"])
        print(f"Password verification result: {is_valid}")
        
        if not is_valid:
            print("Invalid password")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        access_token = create_access_token(data={"sub": str(db_user["_id"])})
        print("Access token created successfully")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        print(f"Login error occurred: {str(e)}")
        raise

@router.get("/hello/")
async def hello_world(user_service: UserService = Depends()):
    return await user_service.hello_world()


@router.get("/identify", dependencies=[Depends(JWTBearer())])
async def identify_user(token: str = Depends(JWTBearer()), user_service: UserService = Depends()):
    try:
        user_id = token.get("sub")
        user_data = await user_service.get_user_by_id(user_id)
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found in database")

        return {
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
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "Invalid token or user data",
                "error": str(e)
            }
        )
