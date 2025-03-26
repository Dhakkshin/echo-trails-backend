from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import json

# Create the FastAPI app instance with proper configuration
app = FastAPI(
    title="Echo Trails API",
    root_path="/api"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a basic test route
@app.get("/")
async def root():
    return {"message": "Hello from Echo Trails API", "status": "ok"}

@app.get("/ping")
async def ping():
    return {"message": "pong"}

# Configure Mangum handler with proper settings for Vercel
handler = Mangum(app, lifespan="off")

# Add this for Vercel compatibility
def app_handler(event, context):
    return handler(event, context)