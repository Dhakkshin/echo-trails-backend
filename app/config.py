# app/config.py
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
MONGO_DETAILS = os.getenv("MONGO_DETAILS")