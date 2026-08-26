"""Aggregation over schema-v1 behavior recordings (replay.py --record).

The numeric core of the action x boss-FSM-state matrix, shared by the CLI
renderer (scripts/analyze_steps.py) and the dashboard's matrix endpoint so
the two can never disagree about the numbers. Read-only: consumes recording
rows, touches no run data.
"""
from collections import Counter
from dataclasses import dataclass

# Direction arrows and verb abbreviations for compact action labels; order
# here fixes label composition (directions first, then verbs).
_DIRS = [("left", "←"), ("right", "→"),
         ("up", "↑"), ("down", "↓")]
_VERBS = [("jump", "Jump"), ("attack", "Atk"), ("dash", "Dash"),
          ("cast", "Cast"), ("focus", "Focus")]


def action_labels(actions: list[dict]) -> list[str]:
    """Compact one-token label per action from its frozen button dict."""
    labels = []
    for buttons in actions:
        parts = [arrow for key, arrow in _DIRS if buttons.get(key)]
        parts += [verb for key, verb in _VERBS if buttons.get(key)]
        labels.append("".join(parts) or "·")
    return labels


# Fixed class order for stable axes and legend colors everywhere.
ACTION_CLASSES = ["focus", "cast", "dash", "attack", "jump", "move", "idle"]


def action_class(buttons: dict) -> str:
    """Collapse a frozen action dict to one class; a combo classifies as
    its highest-priority verb (Jump+Atk is an attack)."""
    for verb in ("focus", "cast", "dash", "attack", "jump"):
        if buttons.get(verb):
            return verb
    if any(buttons.get(d) for d in ("left", "right", "up", "down")):
        return "move"
    return "idle"


def episode_results(rows: list[dict]) -> dict[int, dict]:
    """ep -> episode summary row. Episodes an interrupted recording never
    summarized are simply absent."""
    return {r["ep"]: r for r in rows if r.get("type") == "episode"}


def merge_recordings(recordings: list[list[dict]]) -> tuple[dict, list[dict], int]:
    """(header, concatenated step rows, episode count) across recordings.

    The first file's header speaks for the batch; mixing bosses is refused
    because their FSM state lists (the matrix rows) are different spaces."""
    headers = [r[0] for r in recordings]
    bosses = {h["boss"] for h in headers}
    if len(bosses) > 1:
        raise ValueError(
            f"recordings mix bosses {sorted(bosses)}; the state axis is "
            f"boss-specific, aggregate one boss at a time")
    steps = [row for rec in recordings for row in rec if row["type"] == "step"]
    episodes = sum(1 for rec in recordings for row in rec
                   if row["type"] == "episode")
    return headers[0], steps, episodes


@dataclass
class Aggregate:
    states: list[str]            # observed states, most-visited first
    counts: list[int]            # step count per state, same order
    matrix: list[list[float]]    # mean pi per state (rows sum to ~1)
    modal: list[int]             # most-chosen action id per state
    dropped: list[tuple[str, int]]  # states under min_steps, with counts


def aggregate(rows: list[dict], min_steps: int = 5) -> Aggregate:
    """Fold step rows into the per-state action matrix."""
    pi_sums: dict[str, list[float]] = {}
    chosen: dict[str, Counter] = {}
    for row in rows:
        if row.get("type") != "step":
            continue
        state = row["obs"]["boss_state"]
        pi = row["pi"]
        acc = pi_sums.setdefault(state, [0.0] * len(pi))
        for i, p in enumerate(pi):
            acc[i] += p
        chosen.setdefault(state, Counter())[row["a"]] += 1
    counts = {s: sum(c.values()) for s, c in chosen.items()}
    kept = sorted((s for s in counts if counts[s] >= min_steps),
                  key=lambda s: -counts[s])
    dropped = sorted(((s, counts[s]) for s in counts if counts[s] < min_steps),
                     key=lambda sc: -sc[1])
    return Aggregate(
        states=kept,
        counts=[counts[s] for s in kept],
        matrix=[[p / counts[s] for p in pi_sums[s]] for s in kept],
        # Ties break toward the smallest action id, deterministically.
        modal=[max(chosen[s].items(), key=lambda kv: (kv[1], -kv[0]))[0]
               for s in kept],
        dropped=dropped)
