# app/config.py
from dotenv import load_dotenv
import os
from datetime import timedelta

load_dotenv()

# JWT and Auth Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-256-bit-secret")  # Change in production
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# MongoDB Settings
MONGO_DETAILS = os.getenv("MONGO_DETAILS")