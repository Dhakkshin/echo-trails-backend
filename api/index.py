from http.server import BaseHTTPRequestHandler
import json
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.main import app as fastapi_app
from fastapi.routing import APIRoute

def get_fastapi_routes():
    routes = {}
    for route in fastapi_app.routes:
        if isinstance(route, APIRoute):
            routes[f"{route.path}:{route.methods}"] = route
    return routes

def match_route(request_path, request_method, routes):
    # Remove /api prefix and ensure leading slash
    api_path = request_path.replace('/api', '')
    if not api_path.startswith('/'):
        api_path = '/' + api_path
    
    # Try to match the exact route with method
    route_key = f"{api_path}:{set([request_method])}"
    return routes.get(route_key)

def response_handler(request):
    if request.path == "/api":
        return {"message": "Hello from Echo Trails API"}
    elif request.path == "/api/ping":
        return {"message": "pong"}
    
    routes = get_fastapi_routes()
    matched_route = match_route(request.path, request.command, routes)
    
    if matched_route:
        try:
            response = matched_route.endpoint()
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
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        request_body = self.rfile.read(content_length)
        
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