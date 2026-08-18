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
from hkrl.bosses import BOSSES
from hkrl.exports import exported_generations
from hkrl.rundata import load_run, scan_runs

PAGE = Path(__file__).with_name("dashboard.html")


class _Handler(BaseHTTPRequestHandler):
    def _local_host(self) -> bool:
        # DNS rebinding makes a page origin same-origin with us after the
        # attacker's DNS entry re-resolves to 127.0.0.1, so the browser's
        # same-origin policy no longer protects a bare fetch() -- checking
        # the Host header the browser sent is what actually pins the
        # request to this server.
        port = self.server.server_address[1]
        return self.headers.get("Host") in (f"127.0.0.1:{port}",
                                            f"localhost:{port}")

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        # Only the /api/ routes are data-bearing; "/" is just the static
        # page and stays unguarded so an odd local setup (e.g. a hostname
        # other than localhost/127.0.0.1) can still load it.
        if path.startswith("/api/") and not self._local_host():
            self.send_error(403, "cross-origin request refused")
            return
        if path in ("/", "/summon"):
            # /summon is the same page; the page JS branches on
            # location.pathname so the embedded fonts aren't duplicated
            # into a second file.
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
                # Registry-driven, so the boss picker never drifts from what
                # the launcher will actually accept. {id, name} pairs sorted
                # by display name; the page renders names, submits ids.
                "bosses": [
                    {"id": b.id, "name": b.display_name}
                    for b in sorted(BOSSES.values(),
                                    key=lambda b: b.display_name)
                ],
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
        if not self._local_host():
            self.send_error(403, "cross-origin request refused")
            return
        if not (self.headers.get("Content-Type") or "").startswith(
                "application/json"):
            self.send_error(415, "expected application/json")
            return
        path = unquote(self.path.split("?", 1)[0])
        try:
            # A non-numeric Content-Length raises plain ValueError, and
            # json.JSONDecodeError is itself a ValueError subclass, so one
            # except clause covers both a malformed header and malformed
            # JSON. Clamp negative lengths to 0 rather than letting them
            # reach rfile.read(-1), which would read until EOF.
            length = max(0, int(self.headers.get("Content-Length", 0)))
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json({"error": "invalid request body"}, status=400)
            return
        try:
            if path == "/api/launch":
                self._json({"run_id": launcher.launch(self.server.root,
                                                      body)})
            elif path == "/api/stop":
                self._json({"stopped":
                            launcher.stop(self.server.root)["run_id"]})
            elif path == "/api/replay":
                # launcher.replay validates run_id/gen (missing or bad -> a
                # ValueError the handler below maps to 400); echo the gen back
                # so the page can label the active-run card without a re-fetch.
                run_id = launcher.replay(self.server.root, body.get("run_id"),
                                         body.get("gen"),
                                         body.get("episodes", 3))
                self._json({"replaying": run_id, "gen": body.get("gen")})
            elif path == "/api/export":
                # A synchronous file copy (launcher.export), so unlike
                # replay there is no detached process to report -- just the
                # export's name for the button label.
                self._json({"exported": launcher.export(
                    self.server.root, body.get("run_id"), body.get("gen"),
                    body.get("name"))})
            elif path == "/api/delete":
                self._json({"trashed":
                            launcher.delete(self.server.root,
                                            body.get("run_id"))})
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
        detail = load_run(run_dir)
        # Exported-ness lives on disk, so the page's Export buttons stay
        # "Exported" across re-renders, reloads, and CLI exports alike.
        exported = exported_generations(self.server.root, run_id)
        for g in detail.get("generations", []):
            g["exported"] = g.get("gen") in exported
        self._json(detail)

    def _json(self, payload, status: int = 200):
        self._send(status, "application/json",
                   json.dumps(payload).encode("utf-8"))

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Everything here is dynamic and polled at a fixed URL (the log and
        # status every 2s). Without this a browser is free to serve a stale
        # cached body, freezing the live view on a new run's first poll.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the trainer's terminal quiet
        pass


def make_server(root, port: int = 9700, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.root = Path(root).expanduser()
    return server
