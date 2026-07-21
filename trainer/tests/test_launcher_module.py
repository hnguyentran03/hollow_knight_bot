import json
import os
import signal
import sys
import time

import pytest

from hkrl import launcher

# A stand-in trainer: prints its argv, then idles until SIGINT, which it
# acknowledges in its output before exiting -- enough to observe spawn
# arguments, detachment, and graceful-stop propagation without any game.
STUB = """\
import signal, sys, time
def bye(sig, frame):
    print("sigint received", flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, bye)
print("stub trainer:", " ".join(sys.argv[1:]), flush=True)
time.sleep(60)
"""


@pytest.fixture()
def stub_trainer(tmp_path, monkeypatch):
    script = tmp_path / "stub_train.py"
    script.write_text(STUB)
    monkeypatch.setattr(launcher, "TRAIN_SCRIPT", script)
    yield script
    active = launcher.status(tmp_path)
    if active:  # never leak a stub past a failed test
        os.killpg(os.getpgid(active["pid"]), signal.SIGKILL)


def wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_status_is_none_with_no_pidfiles(tmp_path):
    assert launcher.status(tmp_path) is None


def test_status_cleans_up_stale_pidfiles(tmp_path):
    # pid 0 is guaranteed-dead input with no reuse race; _alive() must
    # treat it (and any pid <= 0) as stale rather than probe it, since
    # os.kill(0, 0) would signal our own process group and "succeed".
    pidfile = tmp_path / "launcher" / "ghost.pid"
    pidfile.parent.mkdir(parents=True)
    pidfile.write_text(json.dumps(
        {"run_id": "ghost", "pid": 0, "started": 0.0}))
    assert launcher.status(tmp_path) is None
    assert not pidfile.exists()


def test_status_ignores_corrupt_pidfiles(tmp_path):
    pidfile = tmp_path / "launcher" / "junk.pid"
    pidfile.parent.mkdir(parents=True)
    pidfile.write_text("not json")
    assert launcher.status(tmp_path) is None
    assert not pidfile.exists()
