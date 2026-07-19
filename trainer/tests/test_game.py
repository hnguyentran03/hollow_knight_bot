import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hkrl.game import GameProcess, PortInUse


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _listener_launcher(launched):
    """A real-subprocess launcher: each launch is a python child that binds
    the port and sleeps. Asserts the relaunch contract at call time: any
    prior holder of the port must already be dead AND reaped."""

    def launch(port, app, visible):
        for _, prior in launched:
            assert prior.poll() is not None, (
                "relaunch() must terminate and reap the old holder "
                "before launching a replacement"
            )
        code = (
            "import socket, time\n"
            "s = socket.socket()\n"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            f"s.bind(('127.0.0.1', {port}))\n"
            "s.listen(8)  # backlog must exceed 1: this fake never accepts, so wait_for_port's already-closed probe connection permanently occupies one backlog slot\n"
            "time.sleep(120)\n"
        )
        proc = subprocess.Popen([sys.executable, "-c", code])
        launched.append((port, proc))
        return proc

    return launch


def _accepts(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2.0):
            return True
    except OSError:
        return False


def test_start_launches_and_waits_for_the_bridge():
    launched = []
    game = GameProcess(port=_free_port(), app=Path("/unused"),
                       launch=_listener_launcher(launched))
    try:
        game.start()
        assert len(launched) == 1
        assert _accepts(game.port)
    finally:
        game.stop()
    assert launched[0][1].poll() is not None  # stop() reaped it


def test_relaunch_reaps_the_old_holder_and_rebinds_the_port():
    launched = []
    game = GameProcess(port=_free_port(), app=Path("/unused"),
                       launch=_listener_launcher(launched))
    try:
        game.start()
        first = launched[0][1]

        game.relaunch(0)  # the launcher itself asserts old-dead-before-new

        assert len(launched) == 2
        assert first.poll() is not None
        assert launched[1][1].poll() is None
        deadline = time.monotonic() + 10.0
        while not _accepts(game.port):
            assert time.monotonic() < deadline, "replacement never bound the port"
            time.sleep(0.1)
    finally:
        game.stop()


def test_start_fails_fast_when_an_unmanaged_process_holds_the_port():
    port = _free_port()
    squatter = socket.socket()
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind(("127.0.0.1", port))
    squatter.listen(1)
    launched = []
    try:
        game = GameProcess(port=port, app=Path("/unused"),
                           launch=_listener_launcher(launched))
        with pytest.raises(PortInUse, match=str(port)):
            game.start()
        assert launched == []  # nothing was started against the collision
    finally:
        squatter.close()
