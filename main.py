from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import users
from app.routers import audio

app = FastAPI()

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