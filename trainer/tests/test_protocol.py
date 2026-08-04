import json
import socket
import threading

import pytest

from hkrl.protocol import Connection, ConnectionClosed


def _serve(handler):
    """Start a one-shot TCP server on an ephemeral port; return its port."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        conn, _ = srv.accept()
        handler(conn)
        conn.close()
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def test_connect_reads_hello_and_roundtrips():
    def handler(conn):
        f = conn.makefile("rwb")
        f.write(b'{"type": "hello", "version": 2}\n')
        f.flush()
        line = f.readline()
        msg = json.loads(line)
        assert msg == {"type": "reset"}
        f.write(json.dumps({"type": "state", "done": False}).encode() + b"\n")
        f.flush()

    port = _serve(handler)
    c = Connection(port=port, timeout=5.0)
    c.connect()
    assert c.hello == {"type": "hello", "version": 2}
    c.send({"type": "reset"})
    assert c.recv() == {"type": "state", "done": False}
    c.close()


def test_recv_raises_on_eof():
    def handler(conn):
        f = conn.makefile("wb")
        f.write(b'{"type": "hello", "version": 2}\n')
        f.flush()

    port = _serve(handler)
    c = Connection(port=port, timeout=5.0)
    c.connect()
    with pytest.raises(ConnectionClosed):
        c.recv()
    c.close()
