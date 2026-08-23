#!/usr/bin/env python3
"""Offline analysis of schema-v1 behavior recordings (replay.py --record).

First visualization: the action x boss-FSM-state matrix -- "which buttons,
in response to what". Reads one or more recordings of the same boss and
renders a heatmap PNG: rows are the boss states actually observed, columns
the 21 actions, each cell the mean recorded policy probability
pi(action | state), with a dot marking the action most often CHOSEN in
that state (intent vs behavior in one picture).

Read-only: consumes recording files, touches no run data. Everything it
needs to label the axes travels inside the recording's frozen header.
"""
import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkrl.recording import read_recording  # noqa: E402

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


def render(agg: Aggregate, header: dict, episodes: int, out: Path) -> None:
    """Heatmap PNG: sequential single-hue cells, white cell gaps, selective
    in-cell values, a ringed dot on each row's modal chosen action."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = action_labels(header["actions"])
    m = np.array(agg.matrix) if agg.states else np.zeros((0, len(labels)))
    n_rows = max(len(agg.states), 1)
    fig, ax = plt.subplots(
        figsize=(0.62 * len(labels) + 3.2, 0.42 * n_rows + 2.4), dpi=150)
    fig.patch.set_facecolor("white")

    im = ax.imshow(m, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right",
                  fontsize=8, color="#333")
    ax.set_yticks(range(len(agg.states)),
                  [f"{s}  (n={n})" for s, n in zip(agg.states, agg.counts)],
                  fontsize=8, color="#333")
    # White gaps between cells; frame and ticks recede.
    ax.set_xticks([x - 0.5 for x in range(1, len(labels))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Selective direct labels: only cells with real mass get a number --
    # except the modal cell, where the dot already marks the spot and text
    # underneath it would collide.
    for r, row in enumerate(agg.matrix):
        for c, v in enumerate(row):
            if v >= 0.15 and c != agg.modal[r]:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if v > 0.55 else "#1a3a5c")
    # The modal CHOSEN action per state: dark dot with a white ring so it
    # stays visible on any cell shade.
    if agg.states:
        ax.scatter(agg.modal, range(len(agg.states)), s=22, color="#12314e",
                   edgecolors="white", linewidths=1.2, zorder=3)

    boss = header.get("boss_spec", {}).get("display_name", header.get("boss"))
    total = sum(agg.counts)
    ax.set_title(
        f"{header.get('run_id')} gen {header.get('gen')} · {boss} · "
        f"{episodes} episodes, {total} steps · dot = most-chosen action",
        fontsize=10, color="#222", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("mean π(action | state)", fontsize=8, color="#333")
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)
    if agg.dropped:
        note = ", ".join(f"{s} ({n})" for s, n in agg.dropped)
        fig.text(0.01, 0.01, f"dropped below min-steps: {note}",
                 fontsize=7, color="#777")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recordings", nargs="+", type=Path,
                    help="schema-v1 .jsonl.gz recording file(s), one boss")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <first recording>"
                         ".action_matrix.png)")
    ap.add_argument("--min-steps", type=int, default=5,
                    help="drop states observed fewer than this many times")
    args = ap.parse_args()

    recs = [read_recording(p.expanduser()) for p in args.recordings]
    header, steps, episodes = merge_recordings(recs)
    agg = aggregate(steps, min_steps=args.min_steps)
    out = args.out or args.recordings[0].expanduser().with_suffix(
        "").with_suffix("").parent / (
        args.recordings[0].name.replace(".jsonl.gz", "")
        + ".action_matrix.png")
    render(agg, header, episodes, out)
    print(f"{len(agg.states)} states x {len(header['actions'])} actions "
          f"from {sum(agg.counts)} steps -> {out}", flush=True)


if __name__ == "__main__":
    main()
