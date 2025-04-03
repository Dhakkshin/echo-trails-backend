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

Response (200 OK):

```json
{
  "message": "Hello from Echo Trails API",
  "status": "ok"
}
```

### Authentication Endpoints

#### Register New User

```http
POST /users/register
Content-Type: application/json
```

Request:

```json
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "securepassword"
}
```

Response (201 Created):

```json
{
  "_id": "65f1a2b3c4d5e6f7g8h9i0j1",
  "username": "johndoe",
  "email": "john@example.com",
  "created_at": "2024-03-26T10:00:00.000Z"
}
```

#### User Login

```http
POST /users/login
Content-Type: application/json
```

Request:

```json
{
  "email": "john@example.com",
  "password": "securepassword"
}
```

Response (200 OK):

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### Get User Identity

```http
GET /users/identify
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "status": "success",
  "token_info": {
    "user_id": "65f1a2b3c4d5e6f7g8h9i0j1",
    "issued_at": "2024-03-26T10:00:00.000Z",
    "expires_at": "2024-03-26T10:30:00.000Z",
    "token_valid": true,
    "scopes": []
  },
  "user_data": {
    "id": "65f1a2b3c4d5e6f7g8h9i0j1",
    "username": "johndoe",
    "email": "john@example.com",
    "created_at": "2024-03-26T10:00:00.000Z",
    "followers": ["65f1a2b3c4d5e6f7g8h9i0j2"],
    "following": ["65f1a2b3c4d5e6f7g8h9i0j3"],
    "pending_follow_requests": ["65f1a2b3c4d5e6f7g8h9i0j4"]
  }
}
```

### User Interaction Endpoints

#### Send Follow Request

```http
POST /users/follow/request/{username}
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "message": "Follow request sent to username"
}
```

#### Accept Follow Request

```http
POST /users/follow/accept/{requester_id}
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "message": "Follow request accepted"
}
```

#### Reject Follow Request

```http
POST /users/follow/reject/{requester_id}
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "message": "Follow request rejected"
}
```

#### Get Pending Follow Requests

```http
GET /users/follow/requests/pending
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
[
  {
    "id": "65f1a2b3c4d5e6f7g8h9i0j1",
    "username": "johndoe"
  }
]
```

#### Unfollow User

```http
POST /users/unfollow/{username}
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "message": "Successfully unfollowed username"
}
```

#### Remove Follower

```http
POST /followers/remove/{username}
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "message": "Successfully removed username from your followers"
}
```

#### Get Following Users

```http
GET /users/following
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
[
  {
    "id": "65f1a2b3c4d5e6f7g8h9i0j1",
    "username": "johndoe"
  },
  {
    "id": "65f1a2b3c4d5e6f7g8h9i0j2",
    "username": "janesmith"
  }
]
```

Common Error Responses for Follow Operations:

- 400: Bad Request
  ```json
  {
    "detail": "Cannot follow yourself"
  }
  ```
  ```json
  {
    "detail": "Already following this user"
  }
  ```
  ```json
  {
    "detail": "Follow request already pending"
  }
  ```
- 404: Not Found
  ```json
  {
    "detail": "User not found"
  }
  ```
  ```json
  {
    "detail": "No pending follow request found"
  }
  ```

Additional Error Responses for Unfollow Operations:

```json
{
  "detail": "Not following this user"
}
```

```json
{
  "detail": "This user is not your follower"
}
```

### Audio Endpoints

#### Upload Audio

```http
POST /audio/upload/
Authorization: Bearer <access_token>
Content-Type: multipart/form-data
```

Request Form Fields:

- `file`: Audio file (required)
- `title`: string (required) - Title of the audio recording
- `latitude`: float (required)
- `longitude`: float (required)
- `range`: float (required) - Distance range in meters where audio is discoverable
- `hidden_until`: datetime ISO string (required)

Response (200 OK):

```json
{
  "id": "64f7e123...",
  "title": "Morning Birds",
  "location": {
    "latitude": 11.024196,
    "longitude": 77.0037728
  },
  "range": 100,
  "hidden_until": "2025-03-28T11:21:54.630Z"
}
```

#### List User's Audio Files

```http
GET /audio/user/files
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "audio_files": [
    {
      "_id": "65f1a2b3c4d5e6f7g8h9i0j1",
      "user_id": "65f1a2b3c4d5e6f7g8h9i0j1",
      "title": "Morning Birds",
      "file_name": "recording.mp3",
      "location": {
        "type": "Point",
        "coordinates": [77.5946, 12.9716]
      },
      "range": 100.0,
      "hidden_until": "2024-03-27T10:00:00.000Z",
      "created_at": "2024-03-26T10:00:00.000Z",
      "audio_data": 1024
    }
  ]
}
```

#### Get Audio File Metadata

```http
GET /audio/files/{audio_id}
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "_id": "65f1a2b3c4d5e6f7g8h9i0j1",
  "user_id": "65f1a2b3c4d5e6f7g8h9i0j1",
  "title": "Morning Birds",
  "file_name": "recording.mp3",
  "location": {
    "type": "Point",
    "coordinates": [77.5946, 12.9716]
  },
  "range": 100.0,
  "hidden_until": "2024-03-27T10:00:00Z",
  "created_at": "2024-03-26T10:00:00Z"
}
```

#### Download Audio File

```http
GET /audio/files/{audio_id}/download
Authorization: Bearer <access_token>
```

Response:

- Content-Type: audio/mpeg
- Content-Disposition: attachment; filename="original_filename.mp3"
- Binary audio data stream

#### Get Nearby Audio Files

```http
GET /audio/nearby?latitude=12.9716&longitude=77.5946
Authorization: Bearer <access_token>
```

Parameters:

- `latitude`: float (required) - Current location latitude
- `longitude`: float (required) - Current location longitude

Response (200 OK):

```json
{
  "nearby_files": [
    {
      "_id": "65f1a2b3c4d5e6f7g8h9i0j1",
      "user_id": "65f1a2b3c4d5e6f7g8h9i0j1",
      "title": "Morning Birds",
      "file_name": "recording.mp3",
      "location": {
        "latitude": 12.9716,
        "longitude": 77.5946
      },
      "range": 100.0,
      "distance": 50.25,
      "hidden_until": "2024-03-27T10:00:00Z",
      "created_at": "2024-03-26T10:00:00Z"
    }
  ],
  "location": {
    "latitude": 12.9716,
    "longitude": 77.5946
  }
}
```

Notes:

- Only returns audio files that belong to the authenticated user
- Files are only returned if they are within the specified range
- Distance is returned in meters and rounded to 2 decimal places
- Only returns files where hidden_until date has passed
- Audio data is excluded from the response for performance

#### Delete Audio File

```http
DELETE /audio/files/{audio_id}
Authorization: Bearer <access_token>
```

Response (200 OK):

```json
{
  "message": "Audio file deleted successfully"
}
```

Notes:

- Only the owner of the audio file can delete it
- Returns 404 if file not found
- Returns 403 if user is not authorized to delete the file

## Error Responses

All endpoints may return the following error responses:

```json
{
  "detail": "Error message description"
}
```

Or for 500 errors:

```json
{
  "detail": {
    "message": "Internal server error description",
    "error_type": "ErrorClassName",
    "error": "Detailed error message"
  }
}
```

Common HTTP Status Codes:

- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

## Authentication

Protected endpoints require Bearer token authentication:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

## Environment Setup

Required environment variables in `.env`:

```
JWT_SECRET_KEY=your_secret_key
ALGORITHM=HS256
MONGO_DETAILS=mongodb+srv://username:password@cluster.mongodb.net/
```

## Development Setup

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create `.env` file with required variables
6. Start development server:
   ```bash
   uvicorn main:app --reload
   ```

## Response Models

### User Model

```json
{
  "_id": "65f1a2b3c4d5e6f7g8h9i0j1",
  "username": "johndoe",
  "email": "john@example.com",
  "created_at": "2024-03-26T10:00:00.000Z",
  "followers": ["65f1a2b3c4d5e6f7g8h9i0j2"],
  "following": ["65f1a2b3c4d5e6f7g8h9i0j3"],
  "pending_follow_requests": ["65f1a2b3c4d5e6f7g8h9i0j4"]
}
```
