"""
FloraScan AI Web Server for Railway Deployment
==============================================
Serves static project files (index.html, styles.css, app.js, models, and assets)
listening on the port assigned dynamically by Railway via the PORT environment variable.
"""

import http.server
import os
import socketserver
import sys

PORT = int(os.environ.get("PORT", 8080))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class FloraScanHTTPHandler(http.server.SimpleHTTPRequestHandler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=DIRECTORY, **kwargs)

  def end_headers(self):
    # Add security & caching headers
    self.send_header(
        "Cache-Control", "no-cache, no-store, must-revalidate"
    )
    self.send_header("Pragma", "no-cache")
    self.send_header("Expires", "0")
    self.send_header("X-Content-Type-Options", "nosniff")
    self.send_header("X-Frame-Options", "SAMEORIGIN")
    super().end_headers()

  def do_GET(self):
    # Route root request explicitly to index.html
    if self.path in ("", "/"):
      self.path = "/index.html"
    return super().do_GET()


class ReusableTCPServer(socketserver.TCPServer):
  allow_reuse_address = True


if __name__ == "__main__":
  with ReusableTCPServer(("0.0.0.0", PORT), FloraScanHTTPHandler) as httpd:
    print(
        f"[FloraScan AI] Production server online at http://0.0.0.0:{PORT}",
        flush=True,
    )
    try:
      httpd.serve_forever()
    except KeyboardInterrupt:
      print("\n[FloraScan AI] Shutting down server gracefully.", flush=True)
      sys.exit(0)
