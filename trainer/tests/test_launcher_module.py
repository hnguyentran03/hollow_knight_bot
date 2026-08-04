import json
import os
import signal
import sys
import threading
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


@pytest.fixture()
def stub_replay(tmp_path, monkeypatch):
    # Same stand-in as stub_trainer, pointed at REPLAY_SCRIPT: replay() spawns
    # scripts/replay.py, which we must not run for real (it launches a game).
    script = tmp_path / "stub_replay.py"
    script.write_text(STUB)
    monkeypatch.setattr(launcher, "REPLAY_SCRIPT", script)
    yield script
    active = launcher.status(tmp_path)
    if active:  # never leak a stub past a failed test
        try:
            os.killpg(os.getpgid(active["pid"]), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # A test that monkeypatched status() to a fake pid (e.g. the
            # "refused while active" case spawned nothing) has nothing to reap.
            pass


def _make_checkpoint(tmp_path, run_id="r1", gen=2):
    """A run dir with gen NNNN's two checkpoint files present -- the pair
    replay() requires before it will spawn. Contents are irrelevant here;
    only their existence is checked."""
    from hkrl.generations import checkpoint_paths
    run_dir = tmp_path / "runs" / run_id
    weights, vecnorm = checkpoint_paths(run_dir, gen)
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_text("weights")
    vecnorm.write_text("vecnorm")
    return run_dir


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


def _log(tmp_path, run_id):
    return (tmp_path / "launcher" / f"{run_id}.log")


def test_launch_spawns_a_detached_run_and_reports_it(tmp_path, stub_trainer):
    run_id = launcher.launch(tmp_path, {"run_id": "r1", "instances": 1,
                                        "timesteps": 100})
    assert run_id == "r1"
    active = launcher.status(tmp_path)
    assert active is not None and active["run_id"] == "r1"
    # Detached: its own session, so a dashboard exit cannot take it down.
    assert os.getsid(active["pid"]) != os.getsid(os.getpid())
    # Its stdout lands in the log, argv includes the unattended flag.
    assert wait_for(lambda: "stub trainer:" in
                    _log(tmp_path, "r1").read_text(errors="replace"))
    assert "--auto" in _log(tmp_path, "r1").read_text(errors="replace")


def test_second_launch_is_refused_while_one_is_alive(tmp_path, stub_trainer):
    launcher.launch(tmp_path, {"run_id": "r1"})
    with pytest.raises(RuntimeError):
        launcher.launch(tmp_path, {"run_id": "r2"})


def test_resume_of_a_missing_run_dir_is_refused(tmp_path, stub_trainer):
    with pytest.raises(ValueError):
        launcher.launch(tmp_path, {"mode": "resume", "run_id": "nope"})


def test_stop_delivers_sigint_and_clears_status(tmp_path, stub_trainer):
    launcher.launch(tmp_path, {"run_id": "r1"})
    assert wait_for(lambda: "stub trainer:" in
                    _log(tmp_path, "r1").read_text(errors="replace"))
    stopped = launcher.stop(tmp_path)
    assert stopped["run_id"] == "r1"
    # The stub acknowledges the SIGINT -- proof the signal reached the
    # trainer process inside the (possibly caffeinate-wrapped) group.
    assert wait_for(lambda: "sigint received" in
                    _log(tmp_path, "r1").read_text(errors="replace"))
    assert wait_for(lambda: launcher.status(tmp_path) is None)


def test_stop_with_nothing_running_raises(tmp_path):
    with pytest.raises(RuntimeError):
        launcher.stop(tmp_path)


def test_tail_returns_the_last_lines_of_the_newest_log(tmp_path):
    d = tmp_path / "launcher"
    d.mkdir()
    (d / "old.log").write_text("ancient\n")
    os.utime(d / "old.log", (1, 1))
    (d / "new.log").write_text("\n".join(f"line{i}" for i in range(300)))
    out = launcher.tail(tmp_path, n=5)
    assert out.splitlines() == [f"line{i}" for i in range(295, 300)]


def test_tail_is_none_with_no_logs(tmp_path):
    assert launcher.tail(tmp_path) is None


def test_launch_uses_one_run_id_even_when_omitted(tmp_path, stub_trainer,
                                                    monkeypatch):
    # _validate() defaults a missing run_id from time.strftime(); launch()
    # must compute it exactly once and thread the same value through to
    # the spawned command, not let a second _validate() call mint a new
    # timestamp that desyncs the pidfile from the run the child reports.
    generated = iter(f"TS{i:06d}" for i in (1, 2))
    monkeypatch.setattr(launcher.time, "strftime",
                        lambda *a, **k: next(generated))
    launcher.launch(tmp_path, {})
    active = launcher.status(tmp_path)
    assert active is not None and active["run_id"] == "TS000001"
    assert wait_for(lambda: "--run-id TS000001" in
                    _log(tmp_path, "TS000001").read_text(errors="replace"))


def test_launch_new_refuses_an_existing_run_dir(tmp_path, stub_trainer):
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    with pytest.raises(ValueError):
        launcher.launch(tmp_path, {"run_id": "r1"})


def test_restart_params_merges_request_over_the_aborted_config(tmp_path):
    # The aborted run's config is the base; model-shaping params always come
    # from it, while a step budget the request carries overrides the config's.
    run_dir = tmp_path / "runs" / "r2"
    run_dir.mkdir(parents=True)
    (run_dir / "config.jsonl").write_text(json.dumps(
        {"run_id": "ignored", "timesteps": 40000, "instances": 2,
         "gen_every": 8000, "batch_size": 32, "n_epochs": 7, "n_steps": 512,
         "seed": 5}) + "\n")
    base = {"mode": "resume", "run_id": "r2"}

    p = launcher._restart_params(run_dir, {**base, "timesteps": 90000})
    assert p["mode"] == "new" and p["run_id"] == "r2"  # id from the dir, reused
    over = launcher.command(tmp_path, p, platform="linux")  # no caffeinate wrap
    assert "--run-id" in over and "r2" in over and "--resume" not in over
    assert "90000" in over and "40000" not in over  # request budget wins
    for flag, val in [("--instances", "2"), ("--gen-every", "8000"),
                      ("--batch-size", "32"), ("--n-epochs", "7"),
                      ("--n-steps", "512"), ("--seed", "5")]:
        assert flag in over and val in over

    # With no override in the request, the config's own timesteps is used.
    base_cmd = launcher.command(tmp_path, launcher._restart_params(run_dir, base),
                                platform="linux")
    assert "40000" in base_cmd


def test_resume_without_a_checkpoint_restarts_fresh(tmp_path, stub_trainer):
    # A run aborted before its first generation: config + run dir, no checkpoint.
    run_dir = tmp_path / "runs" / "r2"
    run_dir.mkdir(parents=True)
    (run_dir / "config.jsonl").write_text(json.dumps(
        {"run_id": "r2", "timesteps": 40000, "instances": 1}) + "\n")
    # Resume restarts it: the empty attempt goes to trash and a fresh run under
    # the same id becomes active -- no "no checkpoint to resume from" error.
    run_id = launcher.launch(tmp_path, {"mode": "resume", "run_id": "r2"})
    assert run_id == "r2"
    assert list((tmp_path / "trash").glob("r2-*"))  # empty attempt moved aside
    assert launcher.status(tmp_path)["run_id"] == "r2"


def test_resume_with_a_checkpoint_still_resumes(tmp_path, stub_trainer):
    # A run with a generation resumes normally (a real --resume, not a restart).
    done = tmp_path / "runs" / "r3"
    done.mkdir(parents=True)
    (done / "config.jsonl").write_text("{}\n")
    (done / "generations.jsonl").write_text("{}\n")
    assert launcher.launch(tmp_path, {"mode": "resume", "run_id": "r3"}) == "r3"
    assert not (tmp_path / "trash").exists()  # a real resume trashes nothing


def test_concurrent_launches_only_one_wins(tmp_path, stub_trainer):
    barrier = threading.Barrier(2)
    results = {}

    def go(i):
        barrier.wait()
        try:
            results[i] = ("ok", launcher.launch(tmp_path, {"run_id": f"c{i}"}))
        except RuntimeError as exc:
            results[i] = ("err", str(exc))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    outcomes = [results[i][0] for i in range(2)]
    assert sorted(outcomes) == ["err", "ok"]
    pidfiles = list((tmp_path / "launcher").glob("*.pid"))
    assert len(pidfiles) == 1
    launcher.stop(tmp_path)


def test_command_rejects_run_id_starting_with_dash(tmp_path, stub_trainer):
    with pytest.raises(ValueError):
        launcher.command(tmp_path, {"run_id": "-rf"}, platform="linux")


def test_command_rejects_run_id_too_long(tmp_path, stub_trainer):
    with pytest.raises(ValueError):
        launcher.command(tmp_path, {"run_id": "x" * 65}, platform="linux")


def test_stop_raises_runtime_error_if_process_exits_first(tmp_path,
                                                            monkeypatch):
    # Simulate the check-then-signal race: status() sees the pid as alive,
    # but it exits before killpg() reaches it. getpgid() is what raises
    # ProcessLookupError in that window, so stub that directly rather than
    # sending any real signal against our own process.
    d = tmp_path / "launcher"
    d.mkdir()
    (d / "r1.pid").write_text(json.dumps(
        {"run_id": "r1", "pid": os.getpid(), "started": 0.0}))

    def boom(pid):
        raise ProcessLookupError

    monkeypatch.setattr(launcher.os, "getpgid", boom)
    with pytest.raises(RuntimeError):
        launcher.stop(tmp_path)


def test_replay_spawns_with_the_right_argv_and_pidfile(tmp_path, stub_replay):
    _make_checkpoint(tmp_path, "r1", gen=2)
    run_id = launcher.replay(tmp_path, "r1", gen=2, episodes=5)
    assert run_id == "r1"
    log = _log(tmp_path, "r1")
    assert wait_for(lambda: "stub trainer:" in log.read_text(errors="replace"))
    argv = log.read_text(errors="replace")
    assert "--auto" in argv
    assert "--gen 2" in argv
    assert "--episodes 5" in argv
    assert str(tmp_path / "runs" / "r1") in argv  # --run-dir points at the run
    # The pidfile carries the extra fields the UI reads to say "replaying".
    active = launcher.status(tmp_path)
    assert active["run_id"] == "r1"
    assert active["mode"] == "replay"
    assert active["gen"] == 2
    # Detached, exactly like launch(): its own session survives a dashboard exit.
    assert os.getsid(active["pid"]) != os.getsid(os.getpid())


def test_replay_is_refused_while_a_run_is_active(tmp_path, stub_replay,
                                                  monkeypatch):
    _make_checkpoint(tmp_path, "r1", gen=1)
    monkeypatch.setattr(launcher, "status",
                        lambda root: {"run_id": "busy", "pid": 1, "started": 0})
    with pytest.raises(RuntimeError):
        launcher.replay(tmp_path, "r1", gen=1)


def test_replay_refuses_a_missing_checkpoint(tmp_path, stub_replay):
    # Run dir exists but the generation's files do not -> a clean 400.
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    with pytest.raises(ValueError):
        launcher.replay(tmp_path, "r1", gen=9)


def test_replay_rejects_bad_gen_and_missing_run_id(tmp_path, stub_replay):
    _make_checkpoint(tmp_path, "r1", gen=1)
    for gen in (0, -1, None, "two"):
        with pytest.raises(ValueError):
            launcher.replay(tmp_path, "r1", gen=gen)
    for run_id in (None, "", "../evil"):
        with pytest.raises(ValueError):
            launcher.replay(tmp_path, run_id, gen=1)


def test_replay_is_stoppable_like_a_run(tmp_path, stub_replay):
    _make_checkpoint(tmp_path, "r1", gen=1)
    launcher.replay(tmp_path, "r1", gen=1)
    assert wait_for(lambda: "stub trainer:" in
                    _log(tmp_path, "r1").read_text(errors="replace"))
    stopped = launcher.stop(tmp_path)
    assert stopped["run_id"] == "r1"
    assert wait_for(lambda: "sigint received" in
                    _log(tmp_path, "r1").read_text(errors="replace"))
    assert wait_for(lambda: launcher.status(tmp_path) is None)


def _make_run(tmp_path, run_id, mtime=None):
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    (d / "generations.jsonl").write_text('{"gen": 1, "timestep": 5}\n')
    (d / "checkpoints").mkdir()
    (d / "checkpoints" / "gen_0001.zip").write_text("weights")
    if mtime is not None:
        for p in [d, *d.rglob("*")]:
            os.utime(p, (mtime, mtime))
    return d


def test_delete_moves_the_run_to_trash_intact(tmp_path):
    _make_run(tmp_path, "done-run", mtime=1000.0)
    trashed = launcher.delete(tmp_path, "done-run")
    assert not (tmp_path / "runs" / "done-run").exists()
    dest = tmp_path / "trash" / trashed
    assert dest.is_dir() and trashed.startswith("done-run-")
    # Moved, not destroyed: checkpoints ride along untouched.
    assert (dest / "checkpoints" / "gen_0001.zip").read_text() == "weights"


def test_delete_refuses_the_active_launched_run(tmp_path, monkeypatch):
    _make_run(tmp_path, "busy", mtime=1000.0)
    monkeypatch.setattr(launcher, "status",
                        lambda root: {"run_id": "busy", "pid": 1, "started": 0})
    with pytest.raises(RuntimeError):
        launcher.delete(tmp_path, "busy")
    assert (tmp_path / "runs" / "busy").exists()


def test_delete_refuses_a_recently_active_run(tmp_path):
    # Fresh mtimes = a run something is still writing (e.g. started from a
    # terminal, invisible to the pidfiles); deleting under it would race.
    _make_run(tmp_path, "warm")
    with pytest.raises(RuntimeError):
        launcher.delete(tmp_path, "warm")
    assert (tmp_path / "runs" / "warm").exists()


def test_delete_allows_a_panel_stopped_run_despite_fresh_mtime(tmp_path):
    # The panel stopped this run, so its fresh final-save mtime must not lock
    # it out of deletion the way an unmarked (terminal-trained) run's does.
    _make_run(tmp_path, "warm")  # fresh mtime -> refused without a marker
    launcher._mark_stopped(tmp_path, "warm")
    trashed = launcher.delete(tmp_path, "warm")
    assert trashed.startswith("warm-")
    assert not (tmp_path / "runs" / "warm").exists()
    # The marker is retired with the run it referred to.
    assert not launcher._stop_marker(tmp_path, "warm").exists()


def test_stop_marks_the_run_for_immediate_deletion(tmp_path, stub_trainer):
    launcher.launch(tmp_path, {"run_id": "s1"})
    launcher.stop(tmp_path)
    assert launcher._stop_marker(tmp_path, "s1").exists()


def test_launch_clears_a_stale_stop_marker(tmp_path, stub_trainer):
    # A prior stop left a marker; relaunching the same id means it is active
    # again, so the stale marker must not linger to weaken a later delete guard.
    launcher._mark_stopped(tmp_path, "s2")
    launcher.launch(tmp_path, {"run_id": "s2"})
    assert not launcher._stop_marker(tmp_path, "s2").exists()


def test_delete_rejects_missing_and_bad_run_ids(tmp_path):
    with pytest.raises(ValueError):
        launcher.delete(tmp_path, "nope")
    with pytest.raises(ValueError):
        launcher.delete(tmp_path, "../escape")


def test_delete_requires_a_run_id(tmp_path):
    for bad in (None, ""):
        with pytest.raises(ValueError):
            launcher.delete(tmp_path, bad)


def _fake_boss(monkeypatch, boss_id="testboss"):
    from hkrl.bosses import BOSSES, BossSpec
    monkeypatch.setitem(BOSSES, boss_id, BossSpec(
        id=boss_id, fsm_states=("Idle", "UNKNOWN"),
        arena_center_x=0.0, arena_half_w=1.0, floor_y=0.0, arena_height=1.0))
    return boss_id


def test_command_forwards_boss_on_new_runs(tmp_path, monkeypatch):
    boss_id = _fake_boss(monkeypatch, "testboss")
    cmd = launcher.command(tmp_path, {"mode": "new", "run_id": "r1",
                                      "boss": boss_id},
                           platform="linux")
    assert "--boss" in cmd
    assert cmd[cmd.index("--boss") + 1] == boss_id


def test_command_drops_boss_on_resume(tmp_path, monkeypatch):
    # A resume derives the boss from the run's own config (train.py's
    # resolve_boss); forwarding it would be redundant at best.
    boss_id = _fake_boss(monkeypatch, "testboss")
    cmd = launcher.command(tmp_path, {"mode": "resume", "run_id": "r1",
                                      "boss": boss_id},
                           platform="linux")
    assert "--boss" not in cmd


def test_validate_rejects_an_unknown_boss(tmp_path):
    with pytest.raises(ValueError, match="boss"):
        launcher.command(tmp_path, {"mode": "new", "run_id": "r1",
                                    "boss": "grimm"}, platform="linux")


def test_restart_params_carries_the_boss(tmp_path, monkeypatch):
    boss_id = _fake_boss(monkeypatch, "testboss")
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "config.jsonl").write_text(
        json.dumps({"boss": boss_id, "n_steps": 512}) + "\n")
    params = launcher._restart_params(run_dir, {"mode": "resume",
                                                "run_id": "r1"})
    assert params["boss"] == boss_id
