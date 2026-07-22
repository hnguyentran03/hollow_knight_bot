"""Phase 0 measurement for async resets: how much wall-clock a multi-instance
run loses to sibling freeze.

Today's SubprocVecEnv is lockstep and blocking, and SB3 auto-resets a done env
synchronously inside the step. While one instance runs its multi-second reset
macro, every sibling is stalled in step_wait for the whole span. This module
quantifies that cost so the (higher-risk) async-reset work can be justified or
killed before it is built -- see docs/superpowers/specs/2026-07-21-async-resets-design.md
section 5, Phase 0.
"""
import json
from pathlib import Path

_PREFIX = "resets_"
_SUFFIX = ".jsonl"


def reset_log_path(run_dir, port: int) -> Path:
    """Sidecar this worker appends its reset spans to.

    One file per port -- like the run dir's per-session monitor_*.csv -- so
    subprocess workers never contend on a single file. Small O_APPEND writes
    of one line stay atomic even if two workers share a file, but per-port
    avoids the question entirely.
    """
    return Path(run_dir) / f"{_PREFIX}{port}{_SUFFIX}"


def append_reset_span(path, span_s: float, t: float) -> None:
    """Append one measured reset span (seconds) with a monotonic timestamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"span_s": float(span_s), "t": float(t)}) + "\n")


def read_reset_spans(run_dir) -> list:
    """Every reset span recorded under `run_dir`, merged across worker files.

    A torn (mid-write) final line is skipped, not raised: the trainer may be
    appending while this reads, exactly like rundata.read_jsonl.
    """
    spans = []
    for sidecar in Path(run_dir).glob(f"{_PREFIX}*{_SUFFIX}"):
        for line in sidecar.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                spans.append(float(json.loads(line)["span_s"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return spans


def _last_record(path):
    """Last well-formed JSON object in a .jsonl, or None. A torn final line
    (the trainer may be mid-write) falls back to the prior line."""
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _wallclock_from_monitors(run_dir) -> float:
    """Rollout wall-clock from the VecMonitor CSVs: latest absolute episode
    end minus the earliest session t_start, spanned across resumed sessions.

    Preferred over generations.jsonl because it exists from the first
    completed episode (no checkpoint needed) and covers every episode -- and
    thus every reset -- through the run's end, so it stays aligned with the
    full span set even on an interrupted run. A torn tail row is skipped.
    """
    starts, ends = [], []
    for csv in Path(run_dir).glob("monitor_*.monitor.csv"):
        lines = csv.read_text().splitlines()
        if not lines or not lines[0].startswith("#"):
            continue
        try:
            t_start = float(json.loads(lines[0][1:])["t_start"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        starts.append(t_start)
        for row in lines[2:]:
            parts = row.split(",")
            if len(parts) != 3:
                continue
            try:
                ends.append(t_start + float(parts[2]))
            except ValueError:
                continue
    if not starts or not ends:
        return 0.0
    return max(ends) - min(starts)


def resolve_run_params(run_dir):
    """(n_instances, wallclock_s) for a run directory.

    n_instances comes from config.jsonl's `instances`, falling back to the
    count of per-port reset sidecars (a run that logged resets ran at least
    that many games). wallclock_s is the rollout time to divide the freeze
    against: the VecMonitor CSV span when episodes have been logged (available
    without a checkpoint), else the last generation's session wall-clock from
    generations.jsonl, else 0.0 when neither exists yet.

    Reads the run files directly rather than via hkrl.generations, which pulls
    in stable_baselines3; this module is imported by hkrl.env on the hot path.
    """
    run_dir = Path(run_dir)
    config = _last_record(run_dir / "config.jsonl") or {}
    n = config.get("instances")
    if not n:
        n = sum(1 for _ in run_dir.glob(f"{_PREFIX}*{_SUFFIX}"))
    wall = _wallclock_from_monitors(run_dir)
    if wall <= 0:
        gen = _last_record(run_dir / "generations.jsonl") or {}
        wall = float(gen.get("wall_clock_s", 0.0))
    return int(n or 0), wall


def report_run(run_dir, n_instances=None, wallclock_s=None) -> dict:
    """Phase 0 summary for a run directory: reads the reset sidecars, resolves
    N and rollout wall-clock (either override wins over the run files), and
    returns the summarize_freeze dict."""
    n, wall = resolve_run_params(run_dir)
    if n_instances is not None:
        n = n_instances
    if wallclock_s is not None:
        wall = wallclock_s
    return summarize_freeze(read_reset_spans(run_dir), n, wall)


def freeze_fraction(spans, n_instances: int, wallclock_s: float) -> float:
    """Fraction of rollout wall-clock lost to sibling freeze.

    `spans` are the measured HKEnv.reset() durations (seconds) across the run.
    Each reset of one instance freezes the other N-1 in lockstep for its whole
    span, so the fleet loses (N-1) * span of compute-time per reset; as a
    fraction of the fleet's total wall-clock (N instances * `wallclock_s`) that
    is (N-1)/N * sum(spans) / wallclock_s.

    Returns 0.0 when there is no sibling to freeze (N<=1) or no wall-clock to
    divide by -- both are "no measurable loss", not errors.
    """
    if n_instances <= 1 or wallclock_s <= 0:
        return 0.0
    return (n_instances - 1) / n_instances * sum(spans) / wallclock_s


def summarize_freeze(spans, n_instances: int, wallclock_s: float) -> dict:
    """The Phase 0 gating numbers: reset-span stats plus the freeze fraction.

    `freeze_fraction` is the go/no-go signal -- the fraction of fleet
    wall-clock the async work could recover. The span stats around it say how
    much of that comes from many short resets vs a few long ones.
    """
    spans = list(spans)
    n = len(spans)
    total = sum(spans)
    return {
        "n_resets": n,
        "total_reset_s": total,
        "mean_reset_s": total / n if n else 0.0,
        "max_reset_s": max(spans) if spans else 0.0,
        "n_instances": n_instances,
        "wallclock_s": wallclock_s,
        "freeze_fraction": freeze_fraction(spans, n_instances, wallclock_s),
    }
