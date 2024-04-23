"""
Run with:
    python http_server_stub.py
"""

from http.server import SimpleHTTPRequestHandler, HTTPServer

host = '0.0.0.0'
port = 8000

class MyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Hello, world! Your port is open and the server is responding.")

if __name__ == "__main__":
    server_address = (host, port)
    httpd = HTTPServer(server_address, MyHandler)
    print(f"Server running on {host}:{port}...")
    httpd.serve_forever()
