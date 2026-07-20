import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hkrl.game import GameFleet, GameProcess, PortInUse


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _free_ports(n):
    # Bound simultaneously so the OS cannot hand the same port out twice.
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def _spawn_port_listener(port):
    code = (
        "import socket, time\n"
        "s = socket.socket()\n"
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        f"s.bind(('127.0.0.1', {port}))\n"
        "s.listen(8)  # backlog must exceed 1: this fake never accepts, so wait_for_port's already-closed probe connection permanently occupies one backlog slot\n"
        "time.sleep(120)\n"
    )
    return subprocess.Popen([sys.executable, "-c", code])


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
        proc = _spawn_port_listener(port)
        launched.append((port, proc))
        return proc

    return launch


def _fleet_launcher(launched):
    """_listener_launcher scoped per port: a fleet legitimately launches one
    slot while every other slot's process is alive, so only a prior holder
    of the SAME port must already be dead and reaped."""

    def launch(port, app, visible):
        for p, prior in launched:
            assert p != port or prior.poll() is not None, (
                "relaunch() must terminate and reap the old holder of this "
                "port before launching a replacement"
            )
        proc = _spawn_port_listener(port)
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


def test_fleet_starts_every_instance_and_stops_them_all():
    launched = []
    ports = _free_ports(2)
    fleet = GameFleet(ports, app=Path("/unused"),
                      launch=_fleet_launcher(launched))
    try:
        fleet.start()
        assert [p for p, _ in launched] == ports
        assert all(_accepts(p) for p in ports)
    finally:
        fleet.stop()
    assert all(proc.poll() is not None for _, proc in launched)


def test_fleet_relaunch_replaces_only_the_named_slot():
    launched = []
    ports = _free_ports(2)
    fleet = GameFleet(ports, app=Path("/unused"),
                      launch=_fleet_launcher(launched))
    try:
        fleet.start()
        survivor, replaced = launched[0][1], launched[1][1]

        fleet.relaunch(1)  # the launcher asserts old-dead-before-new per port

        assert len(launched) == 3
        assert launched[2][0] == ports[1]
        assert replaced.poll() is not None
        assert survivor.poll() is None
        deadline = time.monotonic() + 10.0
        while not _accepts(ports[1]):
            assert time.monotonic() < deadline, "replacement never bound the port"
            time.sleep(0.1)
    finally:
        fleet.stop()


def test_fleet_start_reaps_the_partial_fleet_when_a_later_port_is_squatted():
    launched = []
    ports = _free_ports(2)
    squatter = socket.socket()
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind(("127.0.0.1", ports[1]))
    squatter.listen(1)
    try:
        fleet = GameFleet(ports, app=Path("/unused"),
                          launch=_fleet_launcher(launched))
        with pytest.raises(PortInUse, match=str(ports[1])):
            fleet.start()
        # Slot 0 was already spawned when slot 1's collision surfaced; a
        # partial fleet left running would squat its own port, so start()
        # must have reaped it on the way out.
        assert [p for p, _ in launched] == [ports[0]]
        assert launched[0][1].poll() is not None
    finally:
        squatter.close()
