from http.server import BaseHTTPRequestHandler
import json
import sys
import os
from pathlib import Path
import asyncio
from urllib.parse import parse_qs, urlparse

sys.path.append(str(Path(__file__).parent.parent))

from app.main import app as fastapi_app
from fastapi.routing import APIRoute
from fastapi import Request
from fastapi.types import Send, Receive

async def execute_endpoint(endpoint, scope):
    # Create mock Request object
    request = Request(scope, receive=None, send=None)
    
    # Execute endpoint with dependencies
    if asyncio.iscoroutinefunction(endpoint):
        response = await endpoint()
    else:
        response = endpoint()
    return response

def response_handler(request):
    # Handle built-in routes
    if request.path == "/api":
        return {"message": "Hello from Echo Trails API"}
    elif request.path == "/api/ping":
        return {"message": "pong"}
    
    # Handle FastAPI routes
    try:
        # Parse the path and create scope
        url_parts = urlparse(request.path)
        path = url_parts.path.replace('/api', '', 1) or '/'
        
        # Create a minimal ASGI scope
        scope = {
            "type": "http",
            "method": request.command,
            "scheme": "http",
            "server": (request.server.server_name, request.server.server_port),
            "path": path,
            "query_string": url_parts.query.encode(),
            "headers": [],
        }

        # Find matching route
        for route in fastapi_app.routes:
            if route.path == path and request.command in route.methods:
                # Execute the endpoint
                response = asyncio.run(execute_endpoint(route.endpoint, scope))
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