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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkrl.analysis import (  # noqa: E402,F401  (Aggregate re-exported for callers)
    Aggregate, action_labels, aggregate, merge_recordings)
from hkrl.recording import read_recording  # noqa: E402


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
