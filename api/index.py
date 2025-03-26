from http.server import BaseHTTPRequestHandler
import json

def handle_route(path, method):
    routes = {
        "/api": {"message": "Hello from Echo Trails API"},
        "/api/ping": {"message": "pong"},
        "/api/users/hello": {"message": "Hello from users!"},
        "/api/users/register": {"message": "Register endpoint"},
        "/api/users/login": {"message": "Login endpoint"}
    }
    return routes.get(path, {"error": "Not Found"})

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        response_data = handle_route(self.path, "GET")
        self.send_response(200 if "error" not in response_data else 404)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode())

    def do_POST(self):
        response_data = handle_route(self.path, "POST")
        self.send_response(200 if "error" not in response_data else 404)
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