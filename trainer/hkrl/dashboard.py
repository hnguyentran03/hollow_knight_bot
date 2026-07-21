"""Local web dashboard over the run directories.

Serves the single page in dashboard.html, JSON endpoints backed by
hkrl.rundata, and -- via hkrl.launcher, the one module allowed to mutate
anything -- endpoints that start, resume, stop, and tail training runs.
Run directories themselves are still only ever written by train.py, and
the server binds 127.0.0.1 only.
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from hkrl import launcher
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
        elif path == "/api/launcher":
            self._json({
                "active": launcher.status(self.server.root),
                # Mirrors train.py's own defaults so the form and the CLI
                # start from the same place.
                "defaults": {
                    "run_id": time.strftime("%Y%m%d_%H%M%S"),
                    "instances": 1, "timesteps": 500_000,
                    "gen_every": 15_000, "batch_size": 64, "n_epochs": 5,
                },
            })
        elif path == "/api/launcher/log":
            query = parse_qs(urlsplit(self.path).query)
            try:
                n = max(1, min(5000, int(query.get("n", ["200"])[0])))
            except ValueError:
                n = 200
            text = launcher.tail(self.server.root, n)
            if text is None:
                self.send_error(404)
            else:
                self._send(200, "text/plain; charset=utf-8",
                           text.encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        # Mutating endpoints get two cheap guards a read-only page never
        # needed: the Host check stops DNS-rebinding, and the JSON
        # content-type forces a CORS preflight (which we never answer),
        # so a malicious web page cannot fire a plain form POST at the
        # localhost port.
        port = self.server.server_address[1]
        if self.headers.get("Host") not in (f"127.0.0.1:{port}",
                                            f"localhost:{port}"):
            self.send_error(403, "cross-origin request refused")
            return
        if not (self.headers.get("Content-Type") or "").startswith(
                "application/json"):
            self.send_error(415, "expected application/json")
            return
        path = unquote(self.path.split("?", 1)[0])
        try:
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON body"}, status=400)
            return
        try:
            if path == "/api/launch":
                self._json({"run_id": launcher.launch(self.server.root,
                                                      body)})
            elif path == "/api/stop":
                self._json({"stopped":
                            launcher.stop(self.server.root)["run_id"]})
            else:
                self.send_error(404)
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            self._json({"error": str(exc)}, status=409)

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

    def _json(self, payload, status: int = 200):
        self._send(status, "application/json",
                   json.dumps(payload).encode("utf-8"))

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the trainer's terminal quiet
        pass


def make_server(root, port: int = 9700, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.root = Path(root).expanduser()
    return server
