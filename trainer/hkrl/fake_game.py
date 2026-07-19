"""In-process fake of the HKRLBot mod for tests: serves scripted episodes."""
import json
import socket
import threading


def obs(kx=20.0, khp=9, bhp=900, boss_state="Idle", **kw):
    base = {"kx": kx, "ky": 6.0, "kvx": 0.0, "kvy": 0.0, "khp": khp, "soul": 0,
            "on_ground": True, "dashing": False, "invuln": False, "facing_right": True,
            "bx": 30.0, "by": 6.0, "bvx": 0.0, "bvy": 0.0, "bhp": bhp,
            "boss_state": boss_state, "needle_active": False, "nx": 0.0, "ny": 0.0}
    base.update(kw)
    return base


def state(o, done=False, won=False):
    return {"type": "state", "obs": o, "done": done,
            "info": {"won": won, "scene": "GG_Hornet_1", "attempt": 1}}


class FakeGame:
    def __init__(self, episodes):
        self.episodes = [list(ep) for ep in episodes]
        self.port = None
        self._srv = None
        self._thread = None

    def __enter__(self):
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._srv.close()

    def _run(self):
        conn, _ = self._srv.accept()
        try:
            f = conn.makefile("rwb")

            def send(msg):
                f.write(json.dumps(msg).encode() + b"\n")
                f.flush()

            send({"type": "hello", "version": 1})
            ep = None
            while True:
                line = f.readline()
                if not line:
                    return
                msg = json.loads(line)
                if msg["type"] == "reset":
                    ep = self.episodes.pop(0)
                    send(ep.pop(0))
                elif msg["type"] == "action":
                    send(ep.pop(0))
        finally:
            conn.close()
