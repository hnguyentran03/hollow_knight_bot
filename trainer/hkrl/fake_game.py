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
    def __init__(self, episodes, port=0, fail_resets=0):
        self.episodes = [list(ep) for ep in episodes]
        # port=0 (default) binds an ephemeral port, same as before; a caller
        # that needs to stand a fresh fake back up on a specific port (e.g.
        # simulating a relaunch) passes that port explicitly.
        self._requested_port = port
        # The first `fail_resets` reset requests end with the connection
        # dropped instead of a state frame -- the mod's reset-budget expiry
        # while its boot/reset macro is still ratcheting toward the fight.
        # The listener stays up, so a reconnecting client's next reset
        # proceeds, exactly like the real mod.
        self.fail_resets = fail_resets
        # Keepalive pings answered so far, for tests asserting the pinger
        # actually ran (hkrl/protocol.py's Connection keepalive thread).
        self.pings = 0
        self.port = None
        self._srv = None
        self._thread = None
        self._conn = None

    def __enter__(self):
        self._srv = socket.socket()
        # Rebinding to the same port right after a previous FakeGame closed
        # it (the relaunch-on-the-same-port scenario) would otherwise race
        # the OS reclaiming the address.
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", self._requested_port))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        # Closes the listener and, if one is currently accepted, the live
        # connection too -- so a mid-episode __exit__ reproduces a real
        # instance dying: the client's next send/recv fails immediately
        # instead of after a 30s socket timeout. shutdown() is required, not
        # just close(): _serve's makefile("rwb") holds its own reference to
        # the same socket, so close() alone only drops our reference count
        # and the fd stays open -- and the connection alive -- until the
        # serving thread's file object is closed too. shutdown() acts on the
        # OS socket directly, independent of that refcount.
        self._srv.close()
        if self._conn is not None:
            try:
                self._conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._conn.close()

    def _run(self):
        # Mirrors mod/BridgeServer.cs's AcceptLoop: accept connections in a
        # loop for as long as the listener is open, dropping the previous
        # client whenever a new one connects, rather than serving exactly
        # one connection for the fake's whole lifetime.
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return  # listener closed; nothing left to accept
            self._conn = conn
            self._serve(conn)

    def _serve(self, conn):
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
                    if self.fail_resets > 0:
                        self.fail_resets -= 1
                        conn.shutdown(socket.SHUT_RDWR)
                        return
                    ep = self.episodes.pop(0)
                    send(ep.pop(0))
                elif msg["type"] == "action":
                    send(ep.pop(0))
                elif msg["type"] == "ping":
                    # Mirrors the mod's liveness ping handling: answered in
                    # the read slot, never treated as a protocol violation.
                    self.pings += 1
                    send({"type": "pong"})
        except OSError:
            return
        finally:
            conn.close()
