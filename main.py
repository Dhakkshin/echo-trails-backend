from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import users, audio
from app.database.database import Database, create_indexes

app = FastAPI()

@app.on_event("startup")
async def startup_db_client():
    await Database.connect_db()
    await create_indexes()  # Add this line to create required indexes

@app.on_event("shutdown")
async def shutdown_db_client():
    await Database.close_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello from Echo Trails API", "status": "ok"}

# Include the users router with updated prefix
app.include_router(users.router, prefix="/users", tags=["users"])

# Include the audio router with updated prefix
app.include_router(audio.router, prefix="/audio", tags=["audio"])
