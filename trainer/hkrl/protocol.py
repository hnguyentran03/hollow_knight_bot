"""Line-delimited JSON protocol to the HKRLBot mod (protocol v3)."""
import json
import socket
import threading
import time

# The version the mod must greet with ({"type": "hello", "version": N}).
# Bumped 1 -> 2 when the reset request gained a required "boss" id: a v1 mod
# would ignore the field and silently fight Hornet while the trainer builds
# a different boss's observation space -- exactly the mismatch a version
# check exists to catch. Bumped 2 -> 3 when the needle-era obs fields were
# renamed (needle_active/nx/ny -> projectile_active/px/py): a v2 mod would
# send fields the trainer no longer reads, and _flatten would KeyError on
# the renamed keys. The consumer-side check lives in HKEnv.__init__.
PROTOCOL_VERSION = 3


class ConnectionClosed(Exception):
    pass


class Connection:
    """One bridge connection, kept alive across lockstep gaps.

    The mod's socket read has a hard 10s deadline (BridgeServer.cs
    ReadTimeout): a connection that goes quiet for 10s is dropped. At one
    instance the trainer never goes quiet that long outside a PPO update,
    but with several instances a single slot's multi-second episode reset
    blocks the whole lockstep step -- and every OTHER slot's healthy
    connection sits idle for the duration. The mod's reset budget (22.5s)
    is more than double the idle ceiling, so at N>1 each long reset used to
    starve a sibling into a drop, whose recovery reconnect then aborted the
    resetting slot's macro in turn -- a livelock observed live (2026-07-20)
    as the Knight jumping in place while recoveries churned.

    The fix is this class's keepalive thread: whenever nothing has been
    sent for `keepalive` seconds it sends {"type": "ping"}, which the mod
    answers from its normal read slot with {"type": "pong"} (see
    EpisodeManager's ping handling) and recv() filters back out. Sends are
    lock-serialized because the pinger writes concurrently with the owning
    thread; recv() stays single-threaded (the owner is the only reader).
    """

    def __init__(self, host="127.0.0.1", port=9020, timeout=30.0,
                 keepalive: float | None = 3.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.keepalive = keepalive
        self.hello = None
        self._sock = None
        self._file = None
        self._lock = threading.Lock()
        self._last_send = 0.0
        self._stop = None

    def connect(self):
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        file = sock.makefile("rwb")
        with self._lock:
            self._sock = sock
            self._file = file
        self._last_send = time.monotonic()
        self.hello = self.recv()
        if self.keepalive:
            # One pinger per connect; close() retires it, so a
            # close()/connect() reconnect cycle never leaks a thread pinging
            # a dead file object.
            self._stop = threading.Event()
            threading.Thread(target=self._ping_loop, args=(self._stop,),
                             daemon=True).start()

    def send(self, msg):
        with self._lock:
            if self._file is None:
                raise ConnectionClosed("connection is closed")
            self._file.write(json.dumps(msg).encode("utf-8") + b"\n")
            self._file.flush()
            self._last_send = time.monotonic()

    def recv(self):
        # Pongs are filtered here, at the single choke point, so every
        # consumer -- reset, step, and the supervisor's probes alike -- sees
        # only real protocol traffic regardless of how many keepalive
        # replies queued up while the owner was between messages.
        while True:
            try:
                line = self._file.readline()
            except ValueError:
                # readline() on a closed file (e.g., after abort()) raises ValueError
                raise ConnectionClosed("mod closed the connection")
            if not line:
                raise ConnectionClosed("mod closed the connection")
            msg = json.loads(line)
            if msg.get("type") != "pong":
                return msg

    def _ping_loop(self, stop: threading.Event):
        # Woken at half the keepalive interval (capped at 0.5s) so the
        # worst-case quiet gap stays ~1.5x keepalive -- far inside the
        # mod's 10s ceiling at the 3s default, and still meaningful for
        # the fast intervals tests use.
        while not stop.wait(min(0.5, self.keepalive / 2)):
            if time.monotonic() - self._last_send < self.keepalive:
                continue
            try:
                self.send({"type": "ping"})
            except Exception:
                # The connection died between checks. Not this thread's
                # news to break: the owner discovers it on its next
                # send/recv, exactly as it would have without a pinger.
                return

    def abort(self):
        """Unblock any thread blocked in recv() and close.

        shutdown() acts on the OS socket regardless of the file object's
        buffering or reference count, so a reader parked in readline()
        fails immediately instead of at the socket timeout -- the same
        reasoning as FakeGame.__exit__'s shutdown-before-close.

        The read is taken under the lock, pairing with connect()'s locked
        assignment, so abort() can never observe a half-installed
        connection (a `_sock` from a connect() that hasn't finished setting
        `_file` too).
        """
        with self._lock:
            s = self._sock
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.close()

    def close(self):
        if self._stop is not None:
            self._stop.set()
            self._stop = None
        with self._lock:
            f, s = self._file, self._sock
            self._file = None
            self._sock = None
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
        if s is not None:
            s.close()
