"""Schema-v1 behavior recordings: gzipped JSONL written by instrumented
replay (scripts/replay.py --record), consumed by offline analysis.

One file per replay invocation: a self-describing `header` line (which
freezes the boss spec, action table, and obs key order, so a recording
outlives registry changes), then interleaved `step` and `episode` lines.
Every line is flushed as it is written (gzip sync flush), so an interrupt
leaves a parseable prefix; read_recording tolerates the missing trailer.

Full field tables: docs/superpowers/specs/2026-08-22-behavior-capture-design.md.
"""
import gzip
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import sb3_contrib
import stable_baselines3
import torch

from hkrl.bosses import get_boss
from hkrl.env import ACTIONS, DEFAULT_REWARD, OBS_KEYS
from hkrl.protocol import PROTOCOL_VERSION

SCHEMA_VERSION = 1


def recording_path(directory, gen: int) -> Path:
    """<directory>/<UTC yyyymmdd-HHMMSS>_gen<NNNN>.jsonl.gz"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(directory) / f"{stamp}_gen{gen:04d}.jsonl.gz"


class RecordingWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = gzip.open(self.path, "wt", encoding="utf-8")

    def _write(self, record):
        self._f.write(json.dumps(record) + "\n")
        # Sync-flush per line: the compressed prefix on disk stays
        # decompressible if the process dies before close().
        self._f.flush()

    def header(self, **fields):
        self._write({"type": "header", "schema_version": SCHEMA_VERSION,
                     **fields})

    def step(self, **fields):
        self._write({"type": "step", **fields})

    def episode(self, **fields):
        self._write({"type": "episode", **fields})

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_recording(path) -> list[dict]:
    """All complete records in the file. A recording interrupted before
    close() has no gzip trailer; the EOFError that raises is expected, and
    every line flushed before the interrupt is kept."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        try:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        except EOFError:
            pass
    return rows


def build_header(*, run_dir, gen, weights, vecnorm, boss_id, deterministic,
                 episodes, timescale=1.0, headless=False, auto=False) -> dict:
    """Everything a renderer needs, with the label maps frozen in: registry
    or action-table changes after the fact cannot re-label old data."""
    run_dir = Path(run_dir)
    return dict(
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_dir=str(run_dir), run_id=run_dir.name, gen=gen,
        weights=Path(weights).name, vecnorm=Path(vecnorm).name,
        boss=boss_id, boss_spec=asdict(get_boss(boss_id)),
        actions=ACTIONS, obs_keys=OBS_KEYS,
        reward_config=dict(DEFAULT_REWARD),
        deterministic=bool(deterministic), episodes_requested=episodes,
        timescale=timescale, headless=headless, auto=auto,
        protocol_version=PROTOCOL_VERSION,
        versions=dict(sb3_contrib=sb3_contrib.__version__,
                      stable_baselines3=stable_baselines3.__version__,
                      torch=torch.__version__,
                      python=platform.python_version()))
