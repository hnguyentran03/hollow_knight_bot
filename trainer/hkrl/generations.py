"""Per-generation checkpoints: weights, VecNormalize statistics, and a
measured-stats manifest.

A generation is three artifacts written together. The .zip alone is not a
usable policy: the observation/reward statistics it was trained under live
in VecNormalize, so the _vecnorm.pkl travels beside it, and one JSON line in
generations.jsonl records what the generation actually measured. Files are
written before their manifest line, so a manifest entry always denotes a
complete checkpoint (a crash mid-save leaves at worst orphaned files that
latest_checkpoint never selects over a manifest-backed pair).
"""
import json
import time
from pathlib import Path
from typing import Callable, Sequence

from stable_baselines3.common.callbacks import BaseCallback

MANIFEST_NAME = "generations.jsonl"
CHECKPOINT_DIR = "checkpoints"


def checkpoint_paths(run_dir: Path, gen: int) -> tuple[Path, Path]:
    d = Path(run_dir) / CHECKPOINT_DIR
    return d / f"gen_{gen:04d}.zip", d / f"gen_{gen:04d}_vecnorm.pkl"


def last_generation(run_dir: Path) -> int:
    """Highest generation in the manifest; 0 when the run has none yet."""
    manifest = Path(run_dir) / MANIFEST_NAME
    if not manifest.exists():
        return 0
    gens = [json.loads(line)["gen"]
            for line in manifest.read_text().splitlines() if line.strip()]
    return max(gens, default=0)


def latest_checkpoint(run_dir: Path) -> tuple[int, Path, Path]:
    """(gen, weights, vecnorm) of the newest complete checkpoint.

    Walks the manifest newest-first and skips generations whose files are
    missing (manually deleted, or a save that never finished), so a resume
    always gets a loadable pair.
    """
    for gen in range(last_generation(run_dir), 0, -1):
        weights, vecnorm = checkpoint_paths(run_dir, gen)
        if weights.exists() and vecnorm.exists():
            return gen, weights, vecnorm
    raise FileNotFoundError(f"no complete generation checkpoint under {run_dir}")


def summarize_episodes(episodes: Sequence[dict]) -> dict:
    """Aggregate the callback's per-episode records into manifest stats.

    Records are VecMonitor episode dicts ("r", "l") enriched with "won" and
    "boss_damage_frac" (see GenerationCallback._on_step). Missing keys
    default to 0/False so a recovery-severed episode counts as a damageless
    loss rather than crashing the aggregation.
    """
    n = len(episodes)

    def mean(key):
        return sum(float(ep.get(key, 0.0)) for ep in episodes) / n if n else 0.0

    return {
        "episodes": n,
        "mean_reward": mean("r"),
        "win_rate": mean("won"),
        "mean_episode_len": mean("l"),
        "mean_boss_damage": mean("boss_damage_frac"),
    }


def record_generation(run_dir: Path, gen: int, timestep: int, wall_clock_s: float,
                      stats: dict, recoveries: int,
                      save_model: Callable[[str], None],
                      save_vecnorm: Callable[[str], None]) -> Path:
    """Write one generation: both checkpoint files, then the manifest line.

    SB3-free on purpose (savers are injected) so the file layout is testable
    without constructing a model.
    """
    weights, vecnorm = checkpoint_paths(run_dir, gen)
    weights.parent.mkdir(parents=True, exist_ok=True)
    save_model(str(weights))
    save_vecnorm(str(vecnorm))
    line = {"gen": gen, "timestep": int(timestep),
            "wall_clock_s": round(float(wall_clock_s), 1),
            "recoveries": int(recoveries), **stats}
    with (Path(run_dir) / MANIFEST_NAME).open("a") as f:
        f.write(json.dumps(line) + "\n")
    return weights


class GenerationCallback(BaseCallback):
    """Checkpoints a generation every `every_steps` timesteps.

    `vecnorm` is the VecNormalize wrapper whose statistics travel with the
    weights. `supervisor`, when given, contributes its cumulative recovery
    count to the manifest. Numbering continues from the run's manifest so a
    resumed run extends the sequence instead of overwriting it.
    """

    def __init__(self, run_dir: Path, vecnorm, every_steps: int = 15_000,
                 supervisor=None, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.vecnorm = vecnorm
        self.every_steps = every_steps
        self.supervisor = supervisor
        self._gen = last_generation(self.run_dir)
        self._episodes: list[dict] = []
        self._started_at = None
        self._next_at = None

    def _on_training_start(self) -> None:
        self._started_at = time.monotonic()
        self._next_at = self.num_timesteps + self.every_steps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            ep = info.get("episode")
            if ep is None:
                continue
            if info.get("reset_pending"):
                # An isolated-mode placeholder episode (hkrl/async_reset.py):
                # no fight happened; counting it would dilute win_rate and
                # mean_boss_damage with zeros.
                continue
            record = dict(ep)
            # Read from the raw step info, not VecMonitor info_keywords: the
            # supervisor's recovery frames carry neither key, and VecMonitor
            # indexes keywords unconditionally into each done step's info, so
            # a keyword would KeyError on the first recovery of every run. A
            # severed episode therefore counts as a damageless loss --
            # conservative, and bounded by the number of recoveries.
            record["won"] = bool(info.get("won", False))
            record["boss_damage_frac"] = float(info.get("boss_damage_frac", 0.0))
            self._episodes.append(record)
        if self._next_at is not None and self.num_timesteps >= self._next_at:
            self.save_generation()
        return True

    def save_generation(self) -> Path:
        """Save a generation now, schedule or not -- train.py also calls this
        for the final checkpoint on a clean finish, Ctrl-C, or InstanceDown."""
        self._gen += 1
        wall = time.monotonic() - self._started_at if self._started_at else 0.0
        path = record_generation(
            self.run_dir, gen=self._gen, timestep=self.num_timesteps,
            # Session-relative: a resume restarts this clock alongside its
            # own session; cumulative time is reconstructable from the
            # manifest's per-session ramps.
            wall_clock_s=wall,
            stats=summarize_episodes(self._episodes),
            recoveries=getattr(self.supervisor, "recoveries", 0),
            save_model=self.model.save,
            save_vecnorm=self.vecnorm.save,
        )
        self._episodes = []
        self._next_at = self.num_timesteps + self.every_steps
        return path
