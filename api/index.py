from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# Create the FastAPI app instance directly here instead of importing
app = FastAPI()

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

# Configure Mangum handler
handler = Mangum(app)