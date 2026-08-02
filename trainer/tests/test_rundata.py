import json
import os

import pytest

from hkrl.rundata import attribute_generations, load_run, read_jsonl, scan_runs


def _write_manifest(run_dir, lines):
    with (run_dir / "generations.jsonl").open("a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _gen(gen, timestep, wall, **extra):
    line = {"gen": gen, "timestep": timestep, "wall_clock_s": wall,
            "recoveries": 0, "episodes": 5, "mean_reward": 0.0,
            "win_rate": 0.0, "mean_episode_len": 500.0,
            "mean_boss_damage": 0.4}
    line.update(extra)
    return line


def _write_config(run_dir, timesteps=100_000, resumed_from_gen=None,
                  instances=1, boss="hornet1"):
    with (run_dir / "config.jsonl").open("a") as f:
        f.write(json.dumps({"timesteps": timesteps, "run_id": "r",
                            "instances": instances, "boss": boss,
                            "resumed_from_gen": resumed_from_gen,
                            "started_at": "2026-07-20T01:00:00"}) + "\n")


def _write_monitor(run_dir, name, t_start, rows):
    with (run_dir / f"monitor_{name}.monitor.csv").open("w") as f:
        f.write('#{"t_start": %s, "env_id": "None"}\n' % t_start)
        f.write("r,l,t\n")
        for r, l, t in rows:
            f.write(f"{r},{l},{t}\n")


def _touch_all(run_dir, mtime):
    for p in run_dir.iterdir():
        os.utime(p, (mtime, mtime))


def test_read_jsonl_skips_blank_and_torn_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3, "tru')  # crash mid-write
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_of_a_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "absent.jsonl") == []


def test_load_run_carries_generations_and_latest_config(tmp_path):
    _write_config(tmp_path, timesteps=100_000)
    _write_config(tmp_path, timesteps=50_000, resumed_from_gen=1)
    _write_manifest(tmp_path, [_gen(1, 15_000, 1000.0)])
    run = load_run(tmp_path, now=0)
    assert [g["gen"] for g in run["generations"]] == [1]
    # The latest session's config is the one describing the current process.
    assert run["config"]["timesteps"] == 50_000
    assert run["status"]["sessions"] == 2


def test_episodes_merge_monitor_sessions_in_time_order(tmp_path):
    # Second session's file sorts after the first by name but its episodes
    # must interleave by absolute time, not by file.
    _write_monitor(tmp_path, "20260719_1", t_start=1000.0,
                   rows=[(1.0, 100, 50.0), (2.0, 200, 120.0)])
    _write_monitor(tmp_path, "20260719_2", t_start=1100.0,
                   rows=[(3.0, 300, 10.0)])
    run = load_run(tmp_path, now=0)
    assert [(e["r"], e["l"], e["t"]) for e in run["episodes"]] == [
        (1.0, 100, 1050.0), (3.0, 300, 1110.0), (2.0, 200, 1120.0)]


def test_monitor_torn_last_row_is_skipped(tmp_path):
    path = tmp_path / "monitor_x.monitor.csv"
    path.write_text('#{"t_start": 0, "env_id": "None"}\nr,l,t\n1.0,100,5.0\n2.0,20')
    run = load_run(tmp_path, now=0)
    assert [e["r"] for e in run["episodes"]] == [1.0]


def test_liveness_comes_from_file_mtimes(tmp_path):
    _write_config(tmp_path)
    _touch_all(tmp_path, 10_000.0)
    live = load_run(tmp_path, now=10_000.0 + 60)
    stale = load_run(tmp_path, now=10_000.0 + 600)
    assert live["status"]["live"] is True
    assert stale["status"]["live"] is False
    assert stale["status"]["last_activity"] == 10_000.0


def test_steps_per_hour_measures_the_last_session_only(tmp_path):
    # Session 1 ran at 54k/h; the resumed session runs at 60k/h. A drop in
    # the session-relative wall clock marks the boundary.
    _write_manifest(tmp_path, [
        _gen(1, 15_000, 1000.0),
        _gen(2, 30_000, 2000.0),
        _gen(3, 45_000, 900.0),   # resumed session: 15k steps in 900s
    ])
    run = load_run(tmp_path, now=0)
    assert run["status"]["steps_per_hour"] == pytest.approx(60_000)


def test_steps_per_hour_none_without_a_manifest(tmp_path):
    _write_config(tmp_path)
    assert load_run(tmp_path, now=0)["status"]["steps_per_hour"] is None


def test_target_and_eta_for_a_resumed_run(tmp_path):
    # --timesteps is additive on resume: the session's target is the
    # resumed-from generation's timestep plus the new budget.
    _write_config(tmp_path, timesteps=100_000)
    _write_config(tmp_path, timesteps=100_000, resumed_from_gen=2)
    _write_manifest(tmp_path, [
        _gen(1, 15_000, 1000.0),
        _gen(2, 30_000, 2000.0),
        _gen(3, 60_000, 1800.0),  # resumed session: 30k steps in 1800s = 60k/h
    ])
    run = load_run(tmp_path, now=0)
    assert run["status"]["timestep"] == 60_000
    assert run["status"]["target_timestep"] == 130_000
    # 70k steps left at 60k/h.
    assert run["status"]["eta_s"] == pytest.approx(70_000 / 60_000 * 3600)


def test_a_config_only_run_has_status_but_no_series(tmp_path):
    _write_config(tmp_path)
    run = load_run(tmp_path, now=0)
    assert run["generations"] == []
    assert run["episodes"] == []
    assert run["status"]["timestep"] == 0
    assert run["status"]["target_timestep"] == 100_000
    assert run["status"]["eta_s"] is None


def test_wins_summed_across_all_generations(tmp_path):
    """win_rate is a per-generation mean, so the run's win total is the sum
    of each generation's rate times its episode count."""
    _write_manifest(tmp_path, [
        _gen(1, 15_000, 1000.0, episodes=20, win_rate=0.1),
        _gen(2, 30_000, 2000.0, episodes=18, win_rate=0.5),
    ])
    status = load_run(tmp_path, now=0)["status"]
    assert status["wins"] == 11
    assert "recoveries" not in status


def test_scan_runs_sorts_by_recent_activity_and_summarizes(tmp_path):
    runs = tmp_path / "runs"
    for name, mtime, damage in [("old", 1000.0, 0.2), ("fresh", 2000.0, 0.6)]:
        d = runs / name
        d.mkdir(parents=True)
        _write_config(d)
        _write_manifest(d, [_gen(1, 15_000, 1000.0, mean_boss_damage=damage)])
        _touch_all(d, mtime)
    (runs / "not_a_run").mkdir()  # no config or manifest: ignored

    found = scan_runs(tmp_path, now=2060.0)
    assert [r["id"] for r in found] == ["fresh", "old"]
    assert found[0]["live"] is True
    assert found[1]["live"] is False
    assert found[0]["timestep"] == 15_000
    assert found[0]["mean_boss_damage"] == 0.6


def test_scan_runs_carries_instances_and_target(tmp_path):
    """The summon page's run rows show a run's shape (instances, current /
    target steps) without a second fetch, so the summary carries them."""
    d = tmp_path / "runs" / "shaped"
    d.mkdir(parents=True)
    _write_config(d, timesteps=100_000, instances=2)
    _write_manifest(d, [_gen(1, 15_000, 1000.0)])
    found = scan_runs(tmp_path, now=0)
    assert found[0]["instances"] == 2
    assert found[0]["target_timestep"] == 100_000

    # A pre-feature config without an instances field degrades to None.
    d2 = tmp_path / "runs" / "old-style"
    d2.mkdir(parents=True)
    with (d2 / "config.jsonl").open("a") as f:
        f.write(json.dumps({"timesteps": 50_000, "run_id": "r"}) + "\n")
    found = scan_runs(tmp_path, now=0)
    old = next(r for r in found if r["id"] == "old-style")
    assert old["instances"] is None
    assert old["target_timestep"] == 50_000


def test_scan_runs_carries_the_boss(tmp_path):
    """The summon page's previous-runs row shows each run's boss without a
    second fetch, so the summary carries it -- None for a pre-feature
    config, same degrade-to-None pattern as instances."""
    d = tmp_path / "runs" / "shaped"
    d.mkdir(parents=True)
    _write_config(d, boss="gruz_mother")
    _write_manifest(d, [_gen(1, 15_000, 1000.0)])
    found = scan_runs(tmp_path, now=0)
    assert found[0]["boss"] == "gruz_mother"

    d2 = tmp_path / "runs" / "old-style"
    d2.mkdir(parents=True)
    with (d2 / "config.jsonl").open("a") as f:
        f.write(json.dumps({"timesteps": 50_000, "run_id": "r"}) + "\n")
    found = scan_runs(tmp_path, now=0)
    old = next(r for r in found if r["id"] == "old-style")
    assert old["boss"] is None


def test_scan_runs_of_a_missing_root_is_empty(tmp_path):
    assert scan_runs(tmp_path / "nowhere", now=0) == []


def test_attribute_generations_partitions_by_manifest_counts():
    eps = [{"t": float(i)} for i in range(7)]
    gens = [{"gen": 1, "episodes": 3}, {"gen": 2, "episodes": 2}]
    attribute_generations(eps, gens)
    assert [e["gen"] for e in eps] == [1, 1, 1, 2, 2, None, None]


def test_attribute_generations_empty_manifest_marks_all_in_progress():
    eps = [{"t": 0.0}, {"t": 1.0}]
    attribute_generations(eps, [])
    assert [e["gen"] for e in eps] == [None, None]


def test_attribute_generations_with_fewer_episodes_than_counted():
    # A lost/severed monitor CSV: fewer episodes on disk than the manifest
    # counted. Attribute the prefix in order, no crash.
    eps = [{"t": 0.0}, {"t": 1.0}]
    gens = [{"gen": 1, "episodes": 5}]
    attribute_generations(eps, gens)
    assert [e["gen"] for e in eps] == [1, 1]


def test_load_run_attributes_episode_generations(tmp_path):
    _write_manifest(tmp_path, [_gen(1, 10_000, 100.0, episodes=2)])
    _write_monitor(tmp_path, "a", t_start=1000.0,
                   rows=[(1.0, 100, 10.0), (2.0, 100, 20.0), (3.0, 100, 30.0)])
    run = load_run(tmp_path, now=0)
    assert [e["gen"] for e in run["episodes"]] == [1, 1, None]
