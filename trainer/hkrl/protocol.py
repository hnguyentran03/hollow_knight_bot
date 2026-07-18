"""Line-delimited JSON protocol to the HKRLBot mod (protocol v1)."""
import json
import socket


class ConnectionClosed(Exception):
    pass


class Connection:
    def __init__(self, host="127.0.0.1", port=9020, timeout=30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.hello = None
        self._sock = None
        self._file = None

    def connect(self):
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)
        self._file = self._sock.makefile("rwb")
        self.hello = self.recv()

    def send(self, msg):
        self._file.write(json.dumps(msg).encode("utf-8") + b"\n")
        self._file.flush()

    def recv(self):
        line = self._file.readline()
        if not line:
            raise ConnectionClosed("mod closed the connection")
        return json.loads(line)

    def close(self):
        if self._file is not None:
            self._file.close()
        if self._sock is not None:
            self._sock.close()
