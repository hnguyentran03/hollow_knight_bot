import pathlib
import socket
import subprocess
import sys
import threading
import time

import pytest

# scripts/ is not a package; the existing tests reach it via a path insert
# (see tests/test_random_agent.py). Follow that convention.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import launch_instances  # noqa: E402  (path insert must precede this import)

wait_for_port = launch_instances.wait_for_port
shutdown = launch_instances.shutdown


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


def test_wait_for_port_fails_fast_when_its_process_already_exited():
    # A real child that exits immediately with a distinctive code, standing in
    # for a game that dies on a bad --app path or a mod load failure.
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(17)"])
    proc.wait()

    # Reserved-but-dead port, so nothing can ever accept: without the process
    # check this would block the whole timeout.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 17"):
        wait_for_port(port, timeout=30.0, proc=proc)
    assert time.monotonic() - start < 5.0


def _spawn_sleeper(ignore_sigterm: bool) -> subprocess.Popen:
    """Start a real child that sleeps, optionally ignoring SIGTERM.

    The child prints once its signal handler (or lack thereof) is in place
    and the parent blocks on that line, so the terminate() below can never
    race the handler installation.
    """
    code = (
        "import signal, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_sigterm else "")
        + "print('ready', flush=True)\n"
        + "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    proc.stdout.readline()
    return proc


def test_shutdown_reaps_a_cooperative_child_promptly():
    proc = _spawn_sleeper(ignore_sigterm=False)
    try:
        start = time.monotonic()
        shutdown([proc], grace=5.0)
        elapsed = time.monotonic() - start

        assert proc.poll() is not None
        assert elapsed < 5.0
    finally:
        proc.stdout.close()


def test_shutdown_kills_a_child_that_ignores_sigterm():
    proc = _spawn_sleeper(ignore_sigterm=True)
    try:
        start = time.monotonic()
        shutdown([proc], grace=0.3)
        elapsed = time.monotonic() - start

        # Reaped via SIGKILL escalation, not left to run out its 60s sleep.
        assert proc.poll() is not None
        assert elapsed < 5.0
    finally:
        proc.stdout.close()
