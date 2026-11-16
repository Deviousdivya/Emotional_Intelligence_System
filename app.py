"""Local web interface for the Emotional Intelligence System."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import urllib.parse

from emotion_model import EmotionModel


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
MODEL = EmotionModel(BASE_DIR / "train_data.csv")


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


class AppHandler(BaseHTTPRequestHandler):
    server_version = "StillLocal/1.0"

    def _send(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status, value):
        self._send(status, json_bytes(value), "application/json; charset=utf-8")

    def do_GET(self):
        route = urllib.parse.urlsplit(self.path).path
        if route == "/api/overview":
            self._send_json(
                200,
                {
                    "training_samples": MODEL.state_model.total_rows,
                    "state_counts": MODEL.state_counts,
                    "states": MODEL.state_model.classes,
                    "features": ["journal text", "sleep", "energy", "stress", "session context"],
                },
            )
            return

        if route == "/":
            page = (TEMPLATE_DIR / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return

        if route.startswith("/static/"):
            relative = route.removeprefix("/static/")
            candidate = (STATIC_DIR / relative).resolve()
            try:
                candidate.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self._send_json(404, {"error": "Not found"})
                return
            if candidate.is_file():
                content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                if content_type.startswith("text/") or content_type in ("application/javascript",):
                    content_type += "; charset=utf-8"
                self._send(200, candidate.read_bytes(), content_type)
                return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if urllib.parse.urlsplit(self.path).path != "/api/analyze":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 128_000:
                raise ValueError("The check-in could not be read. Please try again.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("The check-in details are incomplete.")
            result = MODEL.analyze(payload)
            self._send_json(200, result)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "The check-in could not be read. Please try again."})
        except ValueError as error:
            self._send_json(400, {"error": str(error)})

    def log_message(self, format_string, *args):
        # Do not log request bodies or journal text.
        super().log_message(format_string, *args)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    print(f"Emotional Intelligence System running at http://{host}:{port}")
    print(f"Trained from {MODEL.state_model.total_rows:,} local reflections. Journal text is not saved by the server.")
    try:
        ThreadingHTTPServer((host, port), AppHandler).serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
