# Echo Trails API Documentation

Backend API service for Echo Trails application. Built with FastAPI and MongoDB.

## Base URL

```
https://echo-trails-backend.vercel.app
```

## API Endpoints

### Root Endpoint

```http
GET /
```

Response:

```json
{
  "message": "Hello from Echo Trails API",
  "status": "ok"
}
```

### User Management

#### Register New User

```http
POST /users/register
```

Request Body:

```json
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "securepassword"
}
```

Response:

```json
{
  "_id": "user_id",
  "username": "johndoe",
  "email": "john@example.com",
  "created_at": "2024-03-26T10:00:00.000Z"
}
```

#### User Login

```http
POST /users/login
```

Request Body:

```json
{
  "email": "john@example.com",
  "password": "securepassword"
}
```

Response:

```json
{
  "access_token": "eyJ0eXAiOiJKV...",
  "token_type": "bearer"
}
```

#### Test Endpoint

```http
GET /users/hello
```

Response:

```json
{
  "message": "Hello, this is from user services!"
}
```

### Audio Management

#### Upload Audio File

```http
POST /audio/upload
```

Request:

- Multipart form data
- Requires Bearer token authentication

Form Fields:

```json
{
  "file": "audio_file",
  "latitude": "float",
  "longitude": "float",
  "hidden_until": "datetime"
}
```

Response:

```json
{
  "id": "uploaded_audio_id"
}
```

## Authentication

All protected endpoints require a Bearer token in the Authorization header:

```http
Authorization: Bearer <access_token>
```

## Error Responses

The API returns standard HTTP status codes:

- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Server Error

Error Response Format:

```json
{
  "detail": "Error message here"
}
```

## Environment Variables

Required environment variables in `.env`:

```
SECRET_KEY=your_secret_key
ALGORITHM=HS256
MONGO_DETAILS=your_mongodb_connection_string
```

## Development Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate venv: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Set up `.env` file with required variables
6. Run development server: `uvicorn main:app --reload`
