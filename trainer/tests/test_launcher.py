import pathlib
import socket
import sys
import threading
import time

import pytest

# scripts/ is not a package; the existing tests reach it via a path insert
# (see tests/test_random_agent.py). Follow that convention.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import launch_instances  # noqa: E402  (path insert must precede this import)

wait_for_port = launch_instances.wait_for_port


def test_wait_for_port_returns_once_the_port_accepts():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]

    def listen_soon():
        time.sleep(0.2)
        srv.listen(1)

    threading.Thread(target=listen_soon, daemon=True).start()
    wait_for_port(port, timeout=5.0)  # must not raise
    srv.close()


def test_wait_for_port_times_out_on_a_dead_port():
    # Bind and immediately close so the port is reserved-but-dead.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()

    with pytest.raises(TimeoutError):
        wait_for_port(port, timeout=0.5)
