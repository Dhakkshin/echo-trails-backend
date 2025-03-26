from http.server import BaseHTTPRequestHandler
import json
import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from app.main import app as fastapi_app

async def handle_route_async(path, method, body=None):
    # Remove /api prefix for FastAPI routing
    path = path.replace('/api', '', 1) or '/'
    
    # Find matching route in FastAPI app
    for route in fastapi_app.routes:
        if route.path == path and method in route.methods:
            try:
                response = await route.endpoint()
                return response
            except Exception as e:
                return {"error": str(e)}, 500
    
    return {"error": "Not Found"}, 404

def handle_route(path, method, body=None):
    return asyncio.run(handle_route_async(path, method, body))

class handler(BaseHTTPRequestHandler):
    def parse_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if (content_length > 0):
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