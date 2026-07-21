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


def test_command_builds_a_new_run_argv(tmp_path, stub_trainer):
    cmd = launcher.command(
        tmp_path,
        {"run_id": "r1", "instances": 2, "timesteps": 1000, "seed": 7},
        platform="linux")
    assert cmd[0] == sys.executable
    assert cmd[1] == str(stub_trainer)
    assert "--auto" in cmd
    assert ["--run-id", "r1"] == cmd[cmd.index("--run-id"):][:2]
    assert ["--instances", "2"] == cmd[cmd.index("--instances"):][:2]
    assert ["--seed", "7"] == cmd[cmd.index("--seed"):][:2]
    assert "caffeinate" not in cmd


def test_command_wraps_in_caffeinate_on_macos(tmp_path, stub_trainer):
    cmd = launcher.command(tmp_path, {"run_id": "r1"}, platform="darwin")
    assert cmd[:2] == ["caffeinate", "-dims"]


def test_command_resume_forwards_runtime_flags_only(tmp_path, stub_trainer):
    cmd = launcher.command(
        tmp_path,
        {"mode": "resume", "run_id": "old", "timesteps": 500,
         "instances": 2, "n_steps": 64, "batch_size": 8, "n_epochs": 3,
         "seed": 1},
        platform="linux")
    assert ["--resume", str(tmp_path / "runs" / "old")] \
        == cmd[cmd.index("--resume"):][:2]
    assert "--run-id" not in cmd
    # PPO shape comes from the checkpoint on resume (train.py); forwarding
    # these would be misleading noise in `ps` output.
    for flag in ("--n-steps", "--batch-size", "--n-epochs", "--seed"):
        assert flag not in cmd
    assert "--timesteps" in cmd and "--instances" in cmd


def test_command_rejects_bad_input(tmp_path, stub_trainer):
    for params in [
        {"run_id": "../evil"},
        {"run_id": "r1", "instances": 4},
        {"run_id": "r1", "instances": 0},
        {"run_id": "r1", "timesteps": "lots"},
        {"run_id": "r1", "timesteps": -5},
        {"mode": "sideways", "run_id": "r1"},
    ]:
        with pytest.raises(ValueError):
            launcher.command(tmp_path, params, platform="linux")


def test_command_defaults_run_id_to_a_timestamp(tmp_path, stub_trainer):
    cmd = launcher.command(tmp_path, {}, platform="linux")
    run_id = cmd[cmd.index("--run-id") + 1]
    assert len(run_id) == 15 and run_id[8] == "_"  # YYYYmmdd_HHMMSS
