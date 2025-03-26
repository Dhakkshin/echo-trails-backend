from http.server import BaseHTTPRequestHandler
import json

def response_handler(request):
    if request.path == "/api":
        return {"message": "Hello from Echo Trails API"}
    elif request.path == "/api/ping":
        return {"message": "pong"}
    else:
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