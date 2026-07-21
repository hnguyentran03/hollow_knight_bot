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
