"""Read-only aggregation of run directories for the dashboard.

A run directory is the database: generations.jsonl (per-generation stats),
config.jsonl (one line per training session), and VecMonitor CSVs (one per
session, per-episode rows). This module only ever reads those files -- the
trainer may be appending to them while we parse, so a torn final line is a
normal condition, skipped rather than raised.
"""
import json
import time
from pathlib import Path

# Episodes end at most ~3 minutes apart (the env's step ceiling), so five
# silent minutes means the run is stopped or wedged, not merely mid-episode.
LIVE_WINDOW_S = 300


def read_jsonl(path) -> list[dict]:
    """Parse a .jsonl, skipping blank lines and a torn (mid-write) tail."""
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return []
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _episodes(run_dir: Path) -> list[dict]:
    """All monitor-CSV episodes with absolute end times, merged across
    sessions and sorted -- resumed sessions interleave by time, not by file."""
    episodes = []
    for csv in run_dir.glob("monitor_*.monitor.csv"):
        lines = csv.read_text().splitlines()
        if not lines or not lines[0].startswith("#"):
            continue
        try:
            t_start = float(json.loads(lines[0][1:])["t_start"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        for row in lines[2:]:
            parts = row.split(",")
            if len(parts) != 3:
                continue
            try:
                episodes.append({"r": float(parts[0]), "l": int(parts[1]),
                                 "t": t_start + float(parts[2])})
            except ValueError:
                continue
    episodes.sort(key=lambda e: e["t"])
    return episodes


def attribute_generations(episodes: list[dict], generations: list[dict]) -> None:
    """Stamp each episode with the generation whose stats it fed.

    Each manifest line counts the episodes that were aggregated into it, in
    the same time order the monitor CSVs record, so a cumulative-count walk
    recovers the mapping. Episodes past the manifest's total (fought after
    the latest checkpoint) get None. A session that died before its first
    checkpoint leaves uncounted episodes that shift attribution onto the
    next recorded generation -- accepted, the consumer is a tooltip.
    """
    i = 0
    for gen in generations:
        for _ in range(int(gen.get("episodes", 0))):
            if i >= len(episodes):
                return
            episodes[i]["gen"] = gen["gen"]
            i += 1
    for ep in episodes[i:]:
        ep["gen"] = None


def _last_activity(run_dir: Path) -> float | None:
    paths = [run_dir / "generations.jsonl", run_dir / "config.jsonl",
             *run_dir.glob("monitor_*.monitor.csv")]
    times = [p.stat().st_mtime for p in paths if p.exists()]
    return max(times, default=None)


def _sessions(generations: list[dict]) -> list[list[dict]]:
    """Split manifest lines into per-session runs. wall_clock_s is
    session-relative, so a drop marks a resume boundary."""
    sessions: list[list[dict]] = []
    previous_wall = None
    for gen in generations:
        wall = gen.get("wall_clock_s", 0.0)
        if not sessions or (previous_wall is not None and wall < previous_wall):
            sessions.append([])
        sessions[-1].append(gen)
        previous_wall = wall
    return sessions


def _steps_per_hour(generations: list[dict]) -> float | None:
    """Whole-of-last-session average, so a tiny final-save generation does
    not distort the rate."""
    sessions = _sessions(generations)
    if not sessions:
        return None
    last = sessions[-1][-1]
    start_timestep = sessions[-2][-1]["timestep"] if len(sessions) > 1 else 0
    wall = last.get("wall_clock_s", 0.0)
    if wall <= 0:
        return None
    return (last["timestep"] - start_timestep) / wall * 3600.0


def _wins(generations: list[dict]) -> int:
    """Whole-of-run win total. Each manifest line records the generation's
    win_rate over its episode count, so the count of wins is recoverable
    exactly: rate * episodes is an integer up to float error."""
    return sum(round(g.get("win_rate", 0.0) * g.get("episodes", 0))
               for g in generations)


def _target_timestep(generations: list[dict], config: dict | None) -> int | None:
    """The run's absolute step target: the recorded target_timestep when
    the session wrote one (train.py records it since resume inheritance),
    else reconstructed from the old additive semantics -- the resumed-from
    generation's timestep plus that session's --timesteps budget."""
    if config is None:
        return None
    if config.get("target_timestep") is not None:
        return int(config["target_timestep"])
    if "timesteps" not in config:
        return None
    resumed_from = config.get("resumed_from_gen")
    base = next((g["timestep"] for g in generations
                 if g["gen"] == resumed_from), 0)
    return base + int(config["timesteps"])


def load_run(run_dir, now: float | None = None) -> dict:
    run_dir = Path(run_dir)
    now = time.time() if now is None else now
    generations = read_jsonl(run_dir / "generations.jsonl")
    configs = read_jsonl(run_dir / "config.jsonl")
    config = configs[-1] if configs else None

    timestep = generations[-1]["timestep"] if generations else 0
    rate = _steps_per_hour(generations)
    target = _target_timestep(generations, config)
    eta_s = None
    if rate and target is not None and timestep < target:
        eta_s = (target - timestep) / rate * 3600.0
    last_activity = _last_activity(run_dir)

    episodes = _episodes(run_dir)
    attribute_generations(episodes, generations)

    return {
        "id": run_dir.name,
        "generations": generations,
        "episodes": episodes,
        "config": config,
        "status": {
            "live": last_activity is not None and now - last_activity < LIVE_WINDOW_S,
            "last_activity": last_activity,
            "timestep": timestep,
            "target_timestep": target,
            "steps_per_hour": rate,
            "eta_s": eta_s,
            "wins": _wins(generations),
            "sessions": len(configs),
        },
    }


def scan_runs(root, now: float | None = None) -> list[dict]:
    """Summaries of every run under <root>/runs, most recently active first."""
    now = time.time() if now is None else now
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        return []
    summaries = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if not ((d / "generations.jsonl").exists() or (d / "config.jsonl").exists()):
            continue
        run = load_run(d, now=now)
        latest = run["generations"][-1] if run["generations"] else {}
        config = run["config"] or {}
        summaries.append({
            "id": run["id"],
            "live": run["status"]["live"],
            "last_activity": run["status"]["last_activity"],
            "timestep": run["status"]["timestep"],
            # The summon page's run rows state a run's shape without a
            # second fetch; None where an older config predates the field.
            "instances": config.get("instances"),
            "boss": config.get("boss"),
            "target_timestep": run["status"]["target_timestep"],
            "mean_boss_damage": latest.get("mean_boss_damage"),
            "win_rate": latest.get("win_rate"),
        })
    summaries.sort(key=lambda s: s["last_activity"] or 0, reverse=True)
    return summaries
