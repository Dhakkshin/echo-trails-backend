# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from mangum import Mangum
# from fastapi.middleware.cors import CORSMiddleware
# from app.main import app

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.exception_handler(Exception)
# async def validation_exception_handler(request: Request, exc: Exception):
#     return JSONResponse(
#         status_code=500,
#         content={"message": str(exc)},
#     )

# handler = Mangum(app)

from mangum import Mangum
from fastapi import FastAPI
app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello from Vercel!"}

handler = Mangum(app)