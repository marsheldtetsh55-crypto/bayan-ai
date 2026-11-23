import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import bayan

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        data = open('index.html', 'rb').read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(data)
    def do_POST(self):
        n = int(self.headers['Content-Length'])
        msg = json.loads(self.rfile.read(n))['msg']
        out = json.dumps({'reply': bayan.reply(msg)}, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(out)

print('الخادم يعمل على المنفذ 8000')
HTTPServer(('0.0.0.0', 8000), H).serve_forever()
