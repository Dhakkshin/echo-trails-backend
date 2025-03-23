# app/routers/users.py
from fastapi import APIRouter, HTTPException, Depends, status
from app.models.user import User, UserLogin, UserCreate
from app.auth.auth_utils import create_access_token, verify_password
from app.services.user_service import UserService

router = APIRouter()

@router.post("/register/", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, user_service: UserService = Depends()):
    return await user_service.create_user(user)

@router.post("/login/")
async def login_user(user: UserLogin, user_service: UserService = Depends()):
    db_user = await user_service.get_user_by_email(user.email)
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": str(db_user["_id"])})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/hello/")
async def hello_world(user_service: UserService = Depends()):
    return await user_service.hello_world()