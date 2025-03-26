from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from app.models.user import UserCreate, UserLogin
from app.services.user_service import UserService

def handle_route(path, method, body=None):
    routes = {
        "/api": {"message": "Hello from Echo Trails API"},
        "/api/ping": {"message": "pong"},
        "/api/users/hello": {"message": "Hello from users!"}
    }

    # Handle POST routes separately
    if method == "POST":
        if path == "/api/users/register" and body:
            try:
                user_data = UserCreate(**body)
                return {"message": "User registration (mock)", "data": body}
            except Exception as e:
                return {"error": str(e)}, 400
                
        elif path == "/api/users/login" and body:
            try:
                login_data = UserLogin(**body)
                return {"message": "Login successful (mock)", "token": "mock_token"}
            except Exception as e:
                return {"error": str(e)}, 400

    return routes.get(path, {"error": "Not Found"})

class handler(BaseHTTPRequestHandler):
    def parse_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))
        return None

    def do_GET(self):
        response_data = handle_route(self.path, "GET")
        self.send_response(200 if "error" not in response_data else 404)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode())

    def do_POST(self):
        body = self.parse_body()
        response_data = handle_route(self.path, "POST", body)
        status_code = 200
        
        if isinstance(response_data, tuple):
            response_data, status_code = response_data
            
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()