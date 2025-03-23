# app/database/database.py
import motor.motor_asyncio
from app.config import MONGO_DETAILS #import mongo details from config.py

MONGO_DETAILS = MONGO_DETAILS #use the mongo details from config.py

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client.audio_app_db

user_collection = database.get_collection("users_collection")