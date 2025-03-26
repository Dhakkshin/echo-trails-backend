# app/main.py
from fastapi import FastAPI
from app.routers import users

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Include the users router with updated prefix
app.include_router(users.router, prefix="/users", tags=["users"])