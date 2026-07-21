"""Spawn, watch, and stop training runs on behalf of the dashboard.

The only module in the dashboard stack that mutates anything -- the rest
stays read-only. Bookkeeping (pidfile + console log) lives under
<root>/launcher/, deliberately outside the run directories, which remain
written only by train.py.

Runs are spawned detached (their own session, stdio on a log file) so
they survive the dashboard exiting; a restarted dashboard rediscovers a
live run from its pidfile. At most one launched run can be alive at a
time -- the game instances own the bridge ports, so parallel runs could
not coexist anyway.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"

# Popen handles for children this process spawned, so _alive() can reap
# them once they exit: os.kill(pid, 0) cannot tell a zombie from a live
# process (it succeeds on both), and an unreaped child would stay a
# zombie -- and read as "still running" -- for the dashboard's lifetime.
_children: dict[int, subprocess.Popen] = {}


def _dir(root) -> Path:
    d = Path(root).expanduser() / "launcher"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _alive(pid: int) -> bool:
    if pid <= 0:  # kill(0/-n, 0) would probe a process GROUP, not a pid
        return False
    child = _children.get(pid)
    if child is not None:
        if child.poll() is None:
            return True
        del _children[pid]  # reaped; pidfile cleanup follows in status()
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        # PermissionError means the pid exists but is not ours -- after a
        # reboot the number belongs to a stranger, not to our run.
        return False
    return True


def status(root) -> dict | None:
    """The live launched run ({run_id, pid, started}) or None.

    Reads every pidfile under <root>/launcher/ and deletes the stale ones
    (reboot, hard crash) so they can never wedge the launch form shut.
    """
    active = None
    for pidfile in sorted(_dir(root).glob("*.pid")):
        try:
            rec = json.loads(pidfile.read_text())
            pid = int(rec["pid"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError,
                OSError):
            pidfile.unlink(missing_ok=True)
            continue
        if _alive(pid):
            active = rec
        else:
            pidfile.unlink(missing_ok=True)
    return active


# Flags that shape a fresh model only: on resume, PPO hyperparameters come
# from the checkpoint zip (see train.py), so forwarding them would be
# misleading noise. The runtime topology flags are forwarded always.
_NEW_ONLY = ("n_steps", "batch_size", "n_epochs", "seed")
_ALWAYS = ("instances", "timesteps", "gen_every")
_INT_PARAMS = _ALWAYS + _NEW_ONLY


def _validate(params: dict) -> dict:
    mode = params.get("mode", "new")
    if mode not in ("new", "resume"):
        raise ValueError(f"unknown mode {mode!r}")
    run_id = params.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
    # The same rule the dashboard's GET handler applies: a run id is a
    # directory name, never a path.
    if "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError("run id must be a plain directory name")
    clean = {"mode": mode, "run_id": run_id}
    for key in _INT_PARAMS:
        value = params.get(key)
        if value is None or value == "":
            continue
        try:
            clean[key] = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer") from None
    if not 1 <= clean.get("instances", 1) <= 3:
        raise ValueError("instances must be between 1 and 3")
    if clean.get("timesteps", 1) < 1:
        raise ValueError("timesteps must be positive")
    return clean


def command(root, params: dict, platform: str = sys.platform) -> list[str]:
    """The argv a launch() will spawn.

    Split out from launch() so tests -- and a curious operator reading
    the log -- can see exactly what would run without spawning anything.
    """
    p = _validate(params)
    root = Path(root).expanduser()
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--auto", "--root", str(root)]
    if p["mode"] == "resume":
        cmd += ["--resume", str(root / "runs" / p["run_id"])]
    else:
        cmd += ["--run-id", p["run_id"]]
    for key in (_ALWAYS if p["mode"] == "resume" else _INT_PARAMS):
        if key in p:
            cmd += ["--" + key.replace("_", "-"), str(p[key])]
    if platform == "darwin":
        # -dims: display, idle, disk, system -- the same wrapper the README
        # tells a human to type for an overnight run.
        cmd = ["caffeinate", "-dims"] + cmd
    return cmd


def launch(root, params: dict) -> str:
    """Spawn a detached training run; returns its run id.

    Refuses (RuntimeError) while a launched run is alive: the games own
    the bridge ports, so a second fleet could never come up anyway.
    """
    if status(root) is not None:
        raise RuntimeError("a launched run is already active; stop it "
                           "before starting another")
    p = _validate(params)
    if p["mode"] == "resume" \
            and not (Path(root).expanduser() / "runs" / p["run_id"]).is_dir():
        raise ValueError(f"no run named {p['run_id']!r} to resume")
    d = _dir(root)
    # Pass the already-cleaned dict, not the raw params: command() calls
    # _validate() again (it's public and validates on its own), and a
    # second call on raw params with run_id omitted would mint its own
    # timestamp, desyncing the spawned --run-id from this pidfile's name.
    # _validate() is idempotent on an already-clean dict, so this is safe.
    cmd = command(root, p)
    with (d / f"{p['run_id']}.log").open("ab") as log:
        # start_new_session: the run must not die with the dashboard, and
        # it makes the child a process-group leader so stop() can SIGINT
        # the whole group (caffeinate wrapper included) at once.
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT,
                                 start_new_session=True)
    _children[child.pid] = child
    (d / f"{p['run_id']}.pid").write_text(json.dumps(
        {"run_id": p["run_id"], "pid": child.pid, "started": time.time()}))
    return p["run_id"]


def stop(root) -> dict:
    """SIGINT the active run's process group; returns its record.

    One SIGINT is the graceful path: train.py's handler finishes the
    episode in progress, saves a final generation, and reaps the games.
    """
    active = status(root)
    if active is None:
        raise RuntimeError("no launched run is active")
    # status() only just confirmed the pid was alive; it can still exit in
    # the window between that check and this signal, which would surface
    # as an uncaught ProcessLookupError instead of stop()'s documented
    # "nothing to stop" contract.
    try:
        os.killpg(os.getpgid(active["pid"]), signal.SIGINT)
    except ProcessLookupError:
        raise RuntimeError(
            "run exited before it could be stopped") from None
    return active


def tail(root, n: int = 200) -> str | None:
    """Last n lines of the newest launch log, or None if none exists yet."""
    logs = sorted(_dir(root).glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    lines = logs[-1].read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])
