import os
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


def _fake_game(tmp_path: pathlib.Path, posix_body: str,
               windows_body: str) -> pathlib.Path:
    """Write a directly-executable fake game for the current platform.

    launch() execs its app path as-is, so the stand-in must be something the
    OS itself can run: a shell script on POSIX, a .cmd batch file on Windows
    (CreateProcess runs those via cmd.exe on its own).
    """
    if sys.platform == "win32":
        app = tmp_path / "fake_game.cmd"
        app.write_text(windows_body)
    else:
        app = tmp_path / "fake_game"
        app.write_text(posix_body)
        app.chmod(0o755)
    return app


def _spawn_sleeper(ignore_sigterm: bool) -> subprocess.Popen:
    """Start a real child that sleeps, optionally ignoring SIGTERM.

    The child prints once its signal handler (or lack thereof) is in place
    and the parent blocks on that line, so the terminate() below can never
    race the handler installation.

    On Windows the SIG_IGN registration is accepted but moot -- terminate()
    is TerminateProcess, which nothing can ignore -- so the escalation test
    below passes trivially there; the wait-and-reap contract it asserts is
    the same.
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


@pytest.mark.skipif(sys.platform == "win32",
                    reason="process groups are not observable via getpgid "
                           "on Windows; see the win32 test below")
def test_launch_starts_the_game_in_its_own_process_group(tmp_path):
    # A game in the terminal's process group would receive the operator's
    # Ctrl-C directly and die out from under the supervisor mid-save; the
    # only intended kill path is shutdown().
    app = _fake_game(tmp_path, "#!/bin/sh\nsleep 30\n", "")
    proc = launch_instances.launch(9020, app=app, visible=False)
    try:
        assert os.getpgid(proc.pid) != os.getpgid(os.getpid())
    finally:
        launch_instances.shutdown([proc])


@pytest.mark.skipif(sys.platform != "win32",
                    reason="POSIX runs the real-process test above")
def test_launch_detaches_from_console_ctrl_c_on_windows(monkeypatch, tmp_path):
    # Same intent as the POSIX process-group test: the console's CTRL_C_EVENT
    # broadcast must not reach the game. Group membership of a live process
    # has no cheap observable on Windows, so this asserts the creation flag
    # instead -- the suite's one departure from its no-mocks rule.
    captured = {}
    real_popen = subprocess.Popen

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(launch_instances.subprocess, "Popen", capture)
    app = _fake_game(tmp_path, "", "@echo off\r\nping -n 30 127.0.0.1 > NUL\r\n")
    proc = launch_instances.launch(9020, app=app, visible=False)
    try:
        assert captured["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    finally:
        launch_instances.shutdown([proc])


def test_launch_supplies_steam_launch_context(tmp_path):
    # Executed directly (not by Steam), the Steam build's DRM check finds no
    # launch context, asks Steam to relaunch the game, and quits ~15s after
    # boot -- observed live as exit 0 (or SIGBUS during Mono shutdown) at the
    # title menu. SteamAppId/SteamGameId in the child env is that context.
    out = tmp_path / "env_seen"
    app = _fake_game(
        tmp_path,
        f'#!/bin/sh\necho "$SteamAppId $SteamGameId" > "{out}"\nsleep 30\n',
        f'@echo off\r\necho %SteamAppId% %SteamGameId%> "{out}"\r\n'
        f'ping -n 30 127.0.0.1 > NUL\r\n',
    )
    proc = launch_instances.launch(9020, app=app, visible=False)
    try:
        deadline = time.monotonic() + 5.0
        while not out.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert out.read_text().strip() == "367520 367520"
    finally:
        launch_instances.shutdown([proc])


def test_seed_save_dir_seeds_and_refreshes_the_instance_save_clone(tmp_path):
    master = tmp_path / "unity.Team Cherry.Hollow Knight"
    master.mkdir()
    (master / "user1.dat").write_text("godhome, parked at bench")
    bundle_id = "unity.Team Cherry.Hollow Knight.hkrl9030"

    launch_instances.seed_save_dir(bundle_id, source=master,
                                   app_support=tmp_path)
    clone = tmp_path / bundle_id
    assert (clone / "user1.dat").read_text() == "godhome, parked at bench"

    # A later run must start from the master again: in-run churn in the
    # clone (new files, modified saves) is disposable by design.
    (clone / "user1.dat").write_text("mid-run churn")
    (clone / "user2.dat").write_text("stray new-game slot")
    launch_instances.seed_save_dir(bundle_id, source=master,
                                   app_support=tmp_path)
    assert (clone / "user1.dat").read_text() == "godhome, parked at bench"
    assert not (clone / "user2.dat").exists()


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="app-clone isolation is macOS-only (cp -c, "
                           "PlistBuddy); other platforms have none yet")
def test_prepare_instance_clones_the_app_with_a_per_port_bundle_id(
        tmp_path, monkeypatch):
    # A miniature .app bundle standing in for the real 7.5G one; sign=False
    # because an ad-hoc codesign of a fake bundle fails, and the plist edit
    # is the part under test.
    bundle = tmp_path / "game" / "hollow_knight.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    binary = bundle / "Contents" / "MacOS" / "Hollow Knight"
    binary.write_text("#!/bin/sh\n")
    (bundle / "Contents" / "Info.plist").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>'
        "<key>CFBundleIdentifier</key>"
        "<string>unity.Team Cherry.Hollow Knight</string>"
        "</dict></plist>\n")
    # seed_save_dir inside prepare_instance reads the real master save dir;
    # point it at a stand-in.
    master_save = tmp_path / "unity.Team Cherry.Hollow Knight"
    master_save.mkdir()
    (master_save / "user1.dat").write_text("save")
    monkeypatch.setattr(launch_instances, "APP_SUPPORT", tmp_path)

    # sign=False (codesigning a fake bundle fails) and prefs=False (the
    # real `defaults` domains must not be touched from a unit test).
    out = launch_instances.prepare_instance(9030, app=binary,
                                            root=tmp_path / "instances",
                                            sign=False, prefs=False)

    assert out == (tmp_path / "instances" / "port-9030" /
                   "hollow_knight.app" / "Contents" / "MacOS" / "Hollow Knight")
    assert out.exists()
    plist = out.parents[1] / "Info.plist"
    got = subprocess.run(
        ["/usr/libexec/PlistBuddy", "-c", "Print :CFBundleIdentifier",
         str(plist)], capture_output=True, text=True, check=True)
    assert got.stdout.strip() == "unity.Team Cherry.Hollow Knight.hkrl9030"
    assert (tmp_path / "unity.Team Cherry.Hollow Knight.hkrl9030" /
            "user1.dat").read_text() == "save"
