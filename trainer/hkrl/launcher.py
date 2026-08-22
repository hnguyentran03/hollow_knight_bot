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
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from hkrl.bosses import BOSSES
from hkrl.game import DEFAULT_PORT
from hkrl.generations import checkpoint_paths
from hkrl.rundata import LIVE_WINDOW_S, read_jsonl
from hkrl import exports

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
REPLAY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay.py"

# Popen handles for children this process spawned, so _alive() can reap
# them once they exit: os.kill(pid, 0) cannot tell a zombie from a live
# process (it succeeds on both), and an unreaped child would stay a
# zombie -- and read as "still running" -- for the dashboard's lifetime.
_children: dict[int, subprocess.Popen] = {}

# ThreadingHTTPServer runs each request on its own thread, so a
# double-clicked Start button can send two concurrent POSTs. Both would
# otherwise pass the status() check before either pidfile lands, and the
# loser's write would leave a live, untracked fleet. launch() and stop()
# each take this for their whole body.
_lock = threading.Lock()


def _dir(root) -> Path:
    d = Path(root).expanduser() / "launcher"
    d.mkdir(parents=True, exist_ok=True)
    return d


# A run the panel stopped drops a marker beside its pidfile. Its purpose is to
# tell delete() "this run is on its way down by our own hand", so the fresh
# mtime of the final generation train.py saves on the way out does not read as
# a still-training run and lock the run out of deletion for LIVE_WINDOW_S. The
# marker never ends in ".pid", so status()'s "*.pid" scan ignores it.
def _stop_marker(root, run_id) -> Path:
    return _dir(root) / f"{run_id}.stopped"


def _mark_stopped(root, run_id) -> None:
    _stop_marker(root, run_id).write_text(
        json.dumps({"run_id": run_id, "stopped_at": time.time()}))


def _clear_stop_marker(root, run_id) -> None:
    _stop_marker(root, run_id).unlink(missing_ok=True)


def _trash(root, run_dir: Path) -> Path:
    """Move a run dir under <root>/trash/<name>-<timestamp>; returns the dest
    path. A move, never an rmtree -- see delete()'s note; shared by delete()
    and the checkpoint-less-resume restart so both retire a dir the same way."""
    trash = Path(root).expanduser() / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / f"{run_dir.name}-{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(run_dir), str(dest))
    return dest


def _restart_params(run_dir: Path, request: dict) -> dict:
    """Fresh-start params for an aborted (checkpoint-less) run. Model-shaping
    params (n_steps, batch_size, n_epochs, seed), target_kl, and headless come
    from the config the aborted attempt recorded -- a resume request drops the
    ints -- while the request's own values win where present. Returns a 'new'-mode dict reusing
    the run_id; command()'s _validate() coerces/range-checks it.

    Edge: a config with target_kl null restarts in 'new' mode and picks up
    train.py's fresh-run 0.03 default; a recorded 0.0 carries and keeps the cap off."""
    configs = read_jsonl(run_dir / "config.jsonl")
    cfg = configs[-1] if configs else {}
    params = {"mode": "new", "run_id": run_dir.name}
    for key in _INT_PARAMS + _STR_NEW_ONLY + _FLOAT_ALWAYS:
        value = request.get(key, cfg.get(key))
        if value is not None:
            params[key] = value
    for key in _BOOL_ALWAYS:
        # Restarting in 'new' mode skips train.py's resume inheritance, so
        # the recorded value must be carried here or an aborted headless
        # run would silently restart headed.
        value = request.get(key, cfg.get(key))
        if value is True:
            params[key] = value
    return params


def _alive(pid: int) -> bool:
    if pid <= 0:  # kill(0/-n, 0) would probe a process GROUP, not a pid
        return False
    child = _children.get(pid)
    if child is not None:
        if child.poll() is None:
            return True
        # pop, not del: two poll threads can race the same exited child,
        # and the loser's del would raise KeyError into a 500.
        _children.pop(pid, None)
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
    # "*.pid" already excludes the "*.pid.tmp" staging files launch() writes
    # before its atomic os.replace() -- fnmatch requires the name to end in
    # ".pid", and ".pid.tmp" ends in ".tmp".
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
# misleading noise -- train.py now refuses them outright on resume. The
# _ALWAYS set is forwardable in both modes, but only when the request
# actually sets a value: on resume train.py inherits instances/gen_every
# from the run's config.jsonl and finishes to its recorded target when
# timesteps is absent, so an unset field must stay unset.
_NEW_ONLY = ("n_steps", "batch_size", "n_epochs", "seed")
_ALWAYS = ("instances", "timesteps", "gen_every")
_INT_PARAMS = _ALWAYS + _NEW_ONLY

# String-valued params. boss is new-only for the same reason the model-
# shaping ints are: on resume train.py derives it from the run's recorded
# config (and refuses a conflicting flag), so forwarding it is noise.
_STR_NEW_ONLY = ("boss",)

# Float-valued and forwardable in BOTH modes: train.py accepts --target-kl
# on resume by design (the flag exists to change a resumed run's update
# dynamics), unlike the checkpoint-baked ints above. Same unset-stays-unset
# rule as _ALWAYS.
_FLOAT_ALWAYS = ("target_kl",)

# Boolean and forwardable in BOTH modes, mapped to a bare flag pair. On
# resume train.py inherits the last session's recorded value when the flag
# is absent, so an unset field must stay unset (same rule as _ALWAYS); an
# explicit true forces it on (--headless), and an explicit false forces a
# recorded-headless run back to headed (--no-headless, resume mode only --
# 'new' mode's default is headed, so false just stays omitted there).
# _restart_params carries the recorded value because a checkpoint-less
# restart runs in 'new' mode, outside resume inheritance; a request's
# explicit false beats the recorded value there too.
_BOOL_ALWAYS = ("headless",)


def _validate(params: dict) -> dict:
    mode = params.get("mode", "new")
    if mode not in ("new", "resume"):
        raise ValueError(f"unknown mode {mode!r}")
    run_id = params.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
    # The same rule the dashboard's GET handler applies: a run id is a
    # directory name, never a path.
    if "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError("run id must be a plain directory name")
    # A leading "-" would be smuggled into train.py's own argv parser as a
    # flag instead of a value, which fails fast and silently. A run id also
    # becomes a directory name and a pidfile stem; anything long enough to
    # trip ENAMETOOLONG would otherwise surface as an unhandled OSError.
    if run_id.startswith("-"):
        raise ValueError("run id must not start with '-'")
    if len(run_id) > 64:
        raise ValueError("run id too long (max 64 chars)")
    clean = {"mode": mode, "run_id": run_id}
    for key in _INT_PARAMS:
        value = params.get(key)
        if value is None or value == "":
            continue
        try:
            clean[key] = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer") from None
    for key in _FLOAT_ALWAYS:
        value = params.get(key)
        if value is None or value == "":
            continue
        try:
            clean[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number") from None
        if not math.isfinite(clean[key]):
            raise ValueError(f"{key} must be a finite number")
        if clean[key] < 0:
            raise ValueError(f"{key} must be >= 0")
    for key in _BOOL_ALWAYS:
        value = params.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        clean[key] = value
    boss = params.get("boss")
    if boss not in (None, ""):
        if boss not in BOSSES:
            raise ValueError(
                f"unknown boss {boss!r}; known: {', '.join(sorted(BOSSES))}")
        clean["boss"] = boss
    if not 1 <= clean.get("instances", 1) <= 4:
        raise ValueError("instances must be between 1 and 4")
    if clean.get("timesteps", 1) < 1:
        raise ValueError("timesteps must be positive")
    return clean


def _caffeinate(cmd: list[str], platform: str = sys.platform) -> list[str]:
    """Wrap a spawn in caffeinate on macOS so display/idle/disk/system sleep
    can't suspend the game mid-run (a suspended game holds its port open while
    wedged -- see train.py). The same wrapper the README hands a human for an
    overnight run; shared by launch() and replay() so the platform check lives
    in exactly one place."""
    if platform == "darwin":
        return ["caffeinate", "-dims"] + cmd
    return cmd


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
    keys = (_ALWAYS + _FLOAT_ALWAYS if p["mode"] == "resume"
            else _INT_PARAMS + _STR_NEW_ONLY + _FLOAT_ALWAYS)
    for key in keys:
        if key in p:
            cmd += ["--" + key.replace("_", "-"), str(p[key])]
    for key in _BOOL_ALWAYS:
        if p.get(key) is True:
            cmd += ["--" + key.replace("_", "-")]
        elif p.get(key) is False and p["mode"] == "resume":
            cmd += ["--no-" + key.replace("_", "-")]
    return _caffeinate(cmd, platform)


def launch(root, params: dict) -> str:
    """Spawn a detached training run; returns its run id.

    Refuses (RuntimeError) while a launched run is alive: the games own
    the bridge ports, so a second fleet could never come up anyway.
    """
    with _lock:
        if status(root) is not None:
            raise RuntimeError("a launched run is already active; stop it "
                               "before starting another")
        p = _validate(params)
        root_dir = Path(root).expanduser()
        run_dir = root_dir / "runs" / p["run_id"]
        if p["mode"] == "resume":
            if not run_dir.is_dir():
                raise ValueError(f"no run named {p['run_id']!r} to resume")
            if not (run_dir / "generations.jsonl").exists():
                # Aborted before its first checkpoint: there is nothing to
                # resume FROM (train.py's latest_checkpoint would raise), so
                # restart the run fresh with the config its aborted attempt
                # recorded (overlaid with any params the request set, e.g. a
                # new timesteps budget), reusing the id. Move the empty attempt
                # to trash (recoverable) and continue below as a 'new' run.
                p = _restart_params(run_dir, p)
                _trash(root_dir, run_dir)
        elif run_dir.exists():
            # train.py exits instantly on an existing run dir; catching it
            # here instead of letting the spawn die is the difference
            # between a 400 on the page and a run that silently never
            # started.
            raise ValueError(
                f"run {p['run_id']!r} already exists; resume it or "
                "pick another id")
        # A relaunch means this run is active again, so any panel-stop marker
        # left from a prior stop is stale -- drop it (see delete()).
        _clear_stop_marker(root_dir, p["run_id"])
        d = _dir(root)
        # Pass the already-cleaned dict, not the raw params: command() calls
        # _validate() again (it's public and validates on its own), and a
        # second call on raw params with run_id omitted would mint its own
        # timestamp, desyncing the spawned --run-id from this pidfile's name.
        # _validate() is idempotent on an already-clean dict, so this is
        # safe.
        cmd = command(root, p)
        with (d / f"{p['run_id']}.log").open("ab") as log:
            # start_new_session: the run must not die with the dashboard, and
            # it makes the child a process-group leader so stop() can SIGINT
            # the whole group (caffeinate wrapper included) at once.
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT,
                                     start_new_session=True)
        _children[child.pid] = child
        # Write-then-rename: the page polls status() every 2s from another
        # thread, and a reader landing mid-write would parse a truncated
        # file as corrupt and unlink the pidfile of a run that is very much
        # alive. os.replace() is atomic on both POSIX and Windows.
        pidfile = d / f"{p['run_id']}.pid"
        tmp = d / f"{p['run_id']}.pid.tmp"
        tmp.write_text(json.dumps(
            {"run_id": p["run_id"], "pid": child.pid, "started": time.time()}))
        os.replace(tmp, pidfile)
        return p["run_id"]


def replay(root, run_id, gen, episodes: int = 3,
           platform: str = sys.platform) -> str:
    """Spawn a detached replay of one generation; returns its run id.

    Mirrors launch(): the replay launches its own game and occupies the same
    single active slot a training run would (they own the bridge port and the
    game, so neither can run while the other does). It is refused
    (RuntimeError) while any launched run is alive, and validated up front so
    a bad request is a clean 400 rather than a spawn that dies silently.

    The pidfile carries two extra fields beyond a run's -- mode="replay" and
    the gen -- so the active-run card can read "Replaying generation N".
    status() returns the whole record, and stop() SIGINTs the group exactly as
    it does a run (replay.py --auto handles the signal), so neither needs any
    change.
    """
    if not run_id:
        raise ValueError("run_id is required")
    # Same directory-name discipline launch() applies (path/dash/length).
    run_id = _validate({"run_id": run_id})["run_id"]
    try:
        gen = int(gen)
    except (TypeError, ValueError):
        raise ValueError("gen must be an integer") from None
    if gen < 1:
        raise ValueError("gen must be a positive integer")
    try:
        episodes = int(episodes)
    except (TypeError, ValueError):
        raise ValueError("episodes must be an integer") from None
    if episodes < 1:
        raise ValueError("episodes must be positive")
    root = Path(root).expanduser()
    run_dir = root / "runs" / run_id
    weights, vecnorm = checkpoint_paths(run_dir, gen)
    with _lock:
        # Both the existence check and the status check live inside the lock,
        # like launch()'s filesystem preconditions: a delete() (which also
        # takes _lock) could otherwise trash runs/<id> between an unlocked
        # check and the spawn, leaving us booting a whole game only for
        # load_policy to die on the now-missing checkpoint. Checked before
        # status() so a bad request stays a 400 even when the slot is busy.
        #
        # Both files or nothing: the weights are meaningless without the
        # VecNormalize statistics they were trained under (see replay.py), and
        # catching it here is the difference between a 400 on the page and a
        # spawn that dies the moment it tries to load them.
        if not (weights.exists() and vecnorm.exists()):
            raise ValueError(
                f"generation {gen} of {run_id!r} has no checkpoint to replay")
        if status(root) is not None:
            raise RuntimeError("a launched run is already active; stop it "
                               "before replaying")
        cmd = _caffeinate(
            [sys.executable, str(REPLAY_SCRIPT), "--auto",
             "--root", str(root), "--run-dir", str(run_dir),
             "--gen", str(gen), "--episodes", str(episodes),
             "--port", str(DEFAULT_PORT)],
            platform)
        d = _dir(root)
        with (d / f"{run_id}.log").open("ab") as log:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT,
                                     start_new_session=True)
        _children[child.pid] = child
        # Write-then-rename, atomic like launch()'s: the 2s status() poll must
        # never read a half-written pidfile and unlink a live replay.
        pidfile = d / f"{run_id}.pid"
        tmp = d / f"{run_id}.pid.tmp"
        tmp.write_text(json.dumps(
            {"run_id": run_id, "pid": child.pid, "started": time.time(),
             "mode": "replay", "gen": gen}))
        os.replace(tmp, pidfile)
        return run_id


def export(root, run_id, gen=None, name=None) -> str:
    """Export one generation to <root>/exports/; returns the export name.

    A fast synchronous file copy -- no game, no detached process, no
    pidfile -- so unlike launch()/replay() it needs no active-slot check
    and can run while a training run is live.
    """
    if not run_id:
        raise ValueError("run_id is required")
    run_id = _validate({"run_id": run_id})["run_id"]
    if gen is not None:
        try:
            gen = int(gen)
        except (TypeError, ValueError):
            raise ValueError("gen must be an integer") from None
        if gen < 1:
            raise ValueError("gen must be a positive integer")
    if name:
        # Validate export name to prevent path traversal attacks. Mirror the
        # discipline _validate applies to run_id: no path separators, no
        # traversal sequences, no leading dash or dot, and reasonable length.
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError("export name must be a plain directory name (no path separators or ..)")
        if name.startswith("-") or name.startswith("."):
            raise ValueError("export name must not start with '-' or '.'")
        if len(name) > 64:
            raise ValueError("export name too long (max 64 chars)")
    root = Path(root).expanduser()
    run_dir = root / "runs" / run_id
    if not run_dir.is_dir():
        raise ValueError(f"no run named {run_id!r}")
    try:
        return exports.export_generation(root, run_dir, gen=gen,
                                         name=name or None).name
    except FileNotFoundError as exc:
        # latest_checkpoint on a run with no complete generation: a bad
        # request (dashboard 400), not a server fault.
        raise ValueError(str(exc)) from None


def stop(root) -> dict:
    """SIGINT the active run's process group; returns its record.

    One SIGINT is the graceful path: train.py's handler finishes the
    episode in progress, saves a final generation, and reaps the games.
    """
    with _lock:
        active = status(root)
        if active is None:
            raise RuntimeError("no launched run is active")
        # status() only just confirmed the pid was alive; it can still exit
        # in the window between that check and this signal, which would
        # surface as an uncaught ProcessLookupError instead of stop()'s
        # documented "nothing to stop" contract.
        try:
            os.killpg(os.getpgid(active["pid"]), signal.SIGINT)
        except ProcessLookupError:
            raise RuntimeError(
                "run exited before it could be stopped") from None
        # Record that WE stopped this run: on the way out train.py saves a
        # final generation, whose fresh mtime would otherwise make delete()'s
        # recency guard treat the run as still-training and refuse it for
        # LIVE_WINDOW_S. The marker lets a panel-stopped run be deleted at
        # once, while a terminal-trained run (no marker) stays guarded.
        _mark_stopped(root, active["run_id"])
        return active


def tail(root, n: int = 200) -> str | None:
    """Last n lines of the newest launch log, or None if none exists yet."""
    logs = sorted(_dir(root).glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    lines = logs[-1].read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def delete(root, run_id) -> str:
    """Move a run directory to <root>/trash/<id>-<timestamp>; returns the
    trashed directory's name.

    A trash move, never an rmtree: a run dir holds hours of checkpoints,
    and this project has already lived through one mis-writes-destroyed-
    data incident (see backup_saves). Emptying trash/ stays a deliberate,
    manual act. Refuses the active launched run and any run with fresh
    file activity -- a terminal-started training process is invisible to
    the pidfiles, but it cannot hide its mtimes.
    """
    if not run_id:
        raise ValueError("run_id is required")
    run_id = _validate({"run_id": run_id})["run_id"]
    root = Path(root).expanduser()
    run_dir = root / "runs" / run_id
    with _lock:
        # Everything inside the lock, like launch()/stop(): otherwise two
        # concurrent deletes of one id both pass is_dir(), and the loser
        # rglobs a directory the winner already moved -- an uncaught
        # FileNotFoundError surfacing as a 500 instead of a clean 404.
        if not run_dir.is_dir():
            raise ValueError(f"no run named {run_id!r}")
        active = status(root)
        if active is not None and active.get("run_id") == run_id:
            raise RuntimeError(f"{run_id!r} is the active run; stop it first")
        # The recency guard exists to catch a run a terminal is actively
        # training -- invisible to the pidfiles, but its mtimes are fresh. A
        # run the panel itself stopped carries a stop marker, so its own fresh
        # final-save mtime does not lock it out of deletion; only unmarked
        # runs (terminal-trained, or crashed mid-write) stay guarded.
        # lstat, not stat: a broken symlink under the run dir would make
        # stat() raise instead of reporting the link's own mtime.
        if not _stop_marker(root, run_id).exists():
            newest = max((p.lstat().st_mtime for p in run_dir.rglob("*")),
                         default=run_dir.lstat().st_mtime)
            if time.time() - newest < LIVE_WINDOW_S:
                raise RuntimeError(
                    f"{run_id!r} shows activity in the last {LIVE_WINDOW_S}s; "
                    "if a terminal is training it, stop that first")
        dest = _trash(root, run_dir)
        # Sweep the run's launcher bookkeeping (<id>.log, <id>.stopped, any
        # stale <id>.pid/.pid.tmp) into the same bundle -- moved, never
        # destroyed, like the run dir itself. Only here, not in _trash():
        # the checkpoint-less-resume restart reuses the run id and must
        # keep appending to its existing log. The dot after run_id keeps
        # "run1.*" from matching "run12.log".
        for stray in _dir(root).glob(f"{run_id}.*"):
            bundle = dest / "launcher"
            bundle.mkdir(exist_ok=True)
            shutil.move(str(stray), str(bundle / stray.name))
        return dest.name
