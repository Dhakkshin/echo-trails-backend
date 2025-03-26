from http.server import BaseHTTPRequestHandler
import json
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app as fastapi_app
from fastapi.routing import APIRoute

def get_fastapi_routes():
    routes = {}
    for route in fastapi_app.routes:
        if isinstance(route, APIRoute):
            routes[route.path] = route
    return routes

def response_handler(request):
    # Handle built-in routes
    if request.path == "/api":
        return {"message": "Hello from Echo Trails API"}
    elif request.path == "/api/ping":
        return {"message": "pong"}
    
    # Handle FastAPI routes
    fastapi_routes = get_fastapi_routes()
    api_path = request.path.replace('/api', '')  # Remove /api prefix
    if api_path in fastapi_routes:
        try:
            # Call the FastAPI route endpoint
            endpoint = fastapi_routes[api_path]
            response = endpoint.endpoint()
            return response
        except Exception as e:
            return {"error": str(e)}, 500
    
    return {"error": "Not Found"}, 404

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        response_data, status_code = response_handler(self) if isinstance(response_handler(self), tuple) else (response_handler(self), 200)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        self.wfile.write(json.dumps(response_data).encode())
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()