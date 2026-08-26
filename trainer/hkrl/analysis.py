"""Aggregation over schema-v1 behavior recordings (replay.py --record).

The numeric core of the action x boss-FSM-state matrix, shared by the CLI
renderer (scripts/analyze_steps.py) and the dashboard's matrix endpoint so
the two can never disagree about the numbers. Read-only: consumes recording
rows, touches no run data.
"""
import math
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


def _r4(x: float) -> float:
    return float(f"{x:.4g}")


def _steps_by_episode(rows: list[dict]) -> dict[int, list[dict]]:
    by_ep: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("type") == "step":
            by_ep.setdefault(row["ep"], []).append(row)
    return by_ep


def confidence_trace(rows: list[dict]) -> list[dict]:
    """Per-episode certainty timeline: pi(chosen), normalized entropy,
    V(s), khp, and reward-term events. Interrupted trailing episodes are
    kept (result None) -- a partial fight still traces."""
    results = episode_results(rows)
    out = []
    by_ep = _steps_by_episode(rows)
    for ep in sorted(by_ep):
        steps = by_ep[ep]
        ent_max = math.log(len(steps[0]["pi"]))
        events = []
        for s in steps:
            terms = s.get("r_terms", {})
            if terms.get("knight_hit"):
                events.append({"i": s["i"], "kind": "hit"})
            # The env keys this term "boss_damage" (not "boss_hp_scale");
            # see HKEnv._reward_terms.
            if terms.get("boss_damage", 0) > 0:
                events.append({"i": s["i"], "kind": "dealt"})
        last = steps[-1]
        if last.get("won"):
            events.append({"i": last["i"], "kind": "win"})
        elif last.get("trunc"):
            events.append({"i": last["i"], "kind": "timeout"})
        elif last.get("done"):
            events.append({"i": last["i"], "kind": "death"})
        summary = results.get(ep)
        out.append({
            "ep": ep,
            "result": summary["result"] if summary else None,
            "steps": len(steps),
            "pia": [_r4(s["pi"][s["a"]]) for s in steps],
            "ent": [_r4(min(1.0, s["ent"] / ent_max)) for s in steps],
            "v": [_r4(s["v"]) for s in steps],
            "khp": [s["obs"]["khp"] for s in steps],
            "events": events})
    return out


def arena_occupancy(rows: list[dict], nx: int = 24, ny: int = 12) -> dict:
    """Where the Knight stands, split by outcome. Grids are normalized
    within each outcome (comparable panels regardless of episode counts);
    grid[0] is the TOP arena row. Timeouts group with losses (they didn't
    win) but only true losses mark deaths. Episodes without a summary row
    (interrupted recording) are excluded entirely."""
    spec = rows[0]["boss_spec"]
    x0 = spec["arena_center_x"] - spec["arena_half_w"]
    x1 = spec["arena_center_x"] + spec["arena_half_w"]
    y0 = spec["floor_y"]
    y1 = spec["floor_y"] + spec["arena_height"]
    results = episode_results(rows)

    def bin_of(obs):
        ix = min(nx - 1, max(0, int((obs["kx"] - x0) / (x1 - x0) * nx)))
        iy = min(ny - 1, max(0, int((obs["ky"] - y0) / (y1 - y0) * ny)))
        return ix, ny - 1 - iy          # row 0 = top of the arena

    panels = {k: {"episodes": 0, "steps": 0,
                  "grid": [[0] * nx for _ in range(ny)]}
              for k in ("win", "loss")}
    deaths = []
    for ep, steps in _steps_by_episode(rows).items():
        summary = results.get(ep)
        if summary is None:
            continue
        panel = panels["win" if summary["result"] == "WIN" else "loss"]
        panel["episodes"] += 1
        panel["steps"] += len(steps)
        for s in steps:
            ix, iy = bin_of(s["obs"])
            panel["grid"][iy][ix] += 1
        if summary["result"] == "loss":
            deaths.append(list(bin_of(steps[-1]["obs"])))
    for panel in panels.values():
        total = panel["steps"] or 1
        panel["grid"] = [[_r4(c / total) for c in row]
                         for row in panel["grid"]]
    panels["loss"]["deaths"] = deaths
    return {"nx": nx, "ny": ny, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            **panels}


def postmortems(rows: list[dict], window: int = 75) -> list[dict]:
    """The last `window` steps (~5 s at 15 Hz) of every true loss.
    Timeouts are not deaths and wins need no autopsy."""
    results = episode_results(rows)
    by_ep = _steps_by_episode(rows)
    out = []
    for ep in sorted(by_ep):
        summary = results.get(ep)
        if summary is None or summary["result"] != "loss":
            continue
        steps = by_ep[ep]
        out.append({"ep": ep, "total_steps": len(steps),
                    "killing_terms": steps[-1].get("r_terms", {}),
                    "steps": steps[-window:]})
    return out


_REACTION_BUCKETS = [("near", 0.0, 4.0), ("mid", 4.0, 8.0),
                     ("far", 8.0, float("inf"))]


def reaction_profile(rows: list[dict], window: int = 8) -> dict:
    """Action-class mix in the ~half second after a projectile appears,
    bucketed by how far away it appeared. An onset is active-after-
    inactive within an episode; a step-0 already-active projectile also
    counts (it appeared between frames)."""
    actions = rows[0]["actions"]
    counts = {name: Counter() for name, _, _ in _REACTION_BUCKETS}
    ns = Counter()
    onsets = 0
    for steps in _steps_by_episode(rows).values():
        prev_active = False
        for idx, s in enumerate(steps):
            active = bool(s["obs"]["projectile_active"])
            if active and not prev_active:
                onsets += 1
                dist = abs(s["obs"]["px"] - s["obs"]["kx"])
                name = next(n for n, lo, hi in _REACTION_BUCKETS
                            if lo <= dist < hi)
                ns[name] += 1
                for w in steps[idx:idx + window]:
                    counts[name][action_class(actions[w["a"]])] += 1
            prev_active = active
    buckets = []
    for name, lo, hi in _REACTION_BUCKETS:
        total = sum(counts[name].values()) or 1
        buckets.append({"name": name, "lo": lo, "hi": hi, "n": ns[name],
                        "shares": {c: _r4(counts[name][c] / total)
                                   for c in ACTION_CLASSES}})
    return {"onsets": onsets, "window": window,
            "classes": list(ACTION_CLASSES), "buckets": buckets}


def soul_economy(rows: list[dict]) -> dict:
    """Heal-vs-cast over the HP x SOUL space. Buckets split at the 33-soul
    cast/heal cost; 99 (full) is its own column."""
    actions = rows[0]["actions"]
    edges = [(0, 32), (33, 65), (66, 98), (99, 99)]
    cells = [[{"n": 0, "cast": 0, "focus": 0} for _ in edges]
             for _ in range(9)]
    for row in rows:
        if row.get("type") != "step":
            continue
        khp, soul = row["obs"]["khp"], row["obs"]["soul"]
        if not 1 <= khp <= 9:
            continue          # a terminal khp=0 frame isn't a choice state
        col = next(i for i, (lo, hi) in enumerate(edges) if lo <= soul <= hi)
        cell = cells[khp - 1][col]
        cell["n"] += 1
        cls = action_class(actions[row["a"]])
        if cls == "cast":
            cell["cast"] += 1
        elif cls == "focus":
            cell["focus"] += 1
    for hp_row in cells:
        for cell in hp_row:
            n = cell["n"] or 1
            cell["cast"] = _r4(cell["cast"] / n)
            cell["focus"] = _r4(cell["focus"] / n)
    return {"soul_buckets": ["0–32", "33–65", "66–98", "99"],
            "khp": list(range(1, 10)), "cells": cells}


def action_mix(recordings: list[list[dict]]) -> dict:
    """Per-generation action-class shares across many recordings of one
    run: the training-evolution view. Same-gen recordings merge."""
    bosses = {rec[0]["boss"] for rec in recordings}
    if len(bosses) > 1:
        raise ValueError(
            f"recordings mix bosses {sorted(bosses)}; compare one boss's "
            f"run at a time")
    by_gen: dict[int, dict] = {}
    for rec in recordings:
        g = by_gen.setdefault(rec[0]["gen"],
                              {"actions": rec[0]["actions"],
                               "counter": Counter(), "steps": 0,
                               "episodes": 0})
        for row in rec:
            if row["type"] == "step":
                g["counter"][action_class(g["actions"][row["a"]])] += 1
                g["steps"] += 1
            elif row["type"] == "episode":
                g["episodes"] += 1
    rows = []
    for gen in sorted(by_gen):
        g = by_gen[gen]
        total = g["steps"] or 1
        rows.append({"gen": gen, "episodes": g["episodes"],
                     "steps": g["steps"],
                     "shares": {c: _r4(g["counter"][c] / total)
                                for c in ACTION_CLASSES}})
    return {"classes": list(ACTION_CLASSES),
            "gens": [r["gen"] for r in rows], "rows": rows}


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
