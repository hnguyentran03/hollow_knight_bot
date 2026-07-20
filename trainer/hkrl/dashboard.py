"""Local read-only web dashboard over the run directories.

Serves the single page in dashboard.html plus two JSON endpoints backed by
hkrl.rundata. It only ever reads run files and never touches the game port,
so it is safe to leave up beside a live training run.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from hkrl.rundata import load_run, scan_runs

PAGE = Path(__file__).with_name("dashboard.html")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif path == "/api/runs":
            self._json(scan_runs(self.server.root))
        elif path.startswith("/api/run/"):
            self._run(path[len("/api/run/"):])
        else:
            self.send_error(404)

    def _run(self, run_id):
        # The id is a directory name, never a path: anything with a
        # separator (e.g. an unquoted "../") stays inside runs/ by
        # rejection, not by normalization.
        if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
            self.send_error(404)
            return
        run_dir = Path(self.server.root) / "runs" / run_id
        if not ((run_dir / "generations.jsonl").exists()
                or (run_dir / "config.jsonl").exists()):
            self.send_error(404)
            return
        self._json(load_run(run_dir))

    def _json(self, payload):
        self._send(200, "application/json",
                   json.dumps(payload).encode("utf-8"))

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the trainer's terminal quiet
        pass


def make_server(root, port: int = 9021, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.root = Path(root).expanduser()
    return server
