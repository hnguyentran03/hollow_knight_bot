#!/usr/bin/env python3
"""Offline analysis of schema-v1 behavior recordings (replay.py --record).

The behavioral views as subcommands: `matrix` (action x boss-FSM-state
heatmap -- "which buttons, in response to what"; the default when the
first argument is a recording path, which keeps the original bare
invocation working), `trace` (per-episode confidence timeline),
`postmortem` (the last ~5 s before each death), `reaction` (action mix
after a projectile appears), `soul` (heal-vs-cast over HP x SOUL), and
`actionmix` (class shares across recorded generations).

Read-only: consumes recording files, touches no run data. Everything it
needs to label the axes travels inside the recording's frozen header.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkrl.analysis import (  # noqa: E402,F401  (Aggregate re-exported for callers)
    ACTION_CLASSES, Aggregate, action_class, action_labels, aggregate,
    action_mix, confidence_trace, merge_recordings, postmortems,
    reaction_profile, soul_economy)
from hkrl.recording import read_recording  # noqa: E402

# One color per action class, in ACTION_CLASSES order: the dataviz-skill
# reference categorical sequence (light mode), CVD-validated for adjacent
# pairs on white. The dashboard uses the same sequence's dark-mode steps.
CLASS_COLORS = dict(zip(ACTION_CLASSES,
                        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                         "#e87ba4", "#008300", "#4a3aa7"]))
_EVENT_STYLE = {"hit": ("#e34948", "v"), "dealt": ("#2a78d6", "^"),
                "win": ("#008300", "*"), "death": ("#e34948", "x"),
                "timeout": ("#7a8697", "s")}


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


def _mute_axes(*axes):
    for ax in axes:
        ax.tick_params(labelsize=7, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)


def render_trace(episodes, header, out: Path) -> None:
    """One column per episode: policy panel (pi[a] + normalized entropy)
    over a value panel (V(s) with event markers), x in fight seconds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(len(episodes), 1)
    fig, axes = plt.subplots(2 * n, 1, figsize=(8.5, 3.6 * n), dpi=150,
                             squeeze=False)
    fig.patch.set_facecolor("white")
    for k, ep in enumerate(episodes):
        t = [i / 15 for i in range(ep["steps"])]
        top, bot = axes[2 * k][0], axes[2 * k + 1][0]
        top.plot(t, ep["pia"], color="#2a78d6", lw=1.4, label="π(chosen)")
        top.plot(t, ep["ent"], color="#eda100", lw=1.2, label="entropy")
        top.set_ylim(0, 1.02)
        top.set_title(f"episode {ep['ep']} · {ep['result'] or 'interrupted'}"
                      f" · {ep['steps']} steps", fontsize=9, color="#222")
        top.legend(fontsize=7, frameon=False, loc="upper right")
        bot.plot(t, ep["v"], color="#12314e", lw=1.4, label="V(s)")
        for e in ep["events"]:
            color, marker = _EVENT_STYLE[e["kind"]]
            bot.scatter([e["i"] / 15],
                        [ep["v"][min(e["i"], ep["steps"] - 1)]],
                        s=26, color=color, marker=marker, zorder=3,
                        label=e["kind"])
        bot.set_xlabel("fight time (s)", fontsize=8)
        bot.set_ylabel("V(s)", fontsize=8)
        # Dedup marker legend entries (every hit adds one otherwise).
        handles, labels = bot.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        bot.legend(seen.values(), seen.keys(), fontsize=7, frameon=False,
                   loc="upper right")
        for ax in (top, bot):
            _mute_axes(ax)
            ax.grid(color="#eee", lw=0.6)
    boss = header.get("boss_spec", {}).get("display_name", header.get("boss"))
    fig.suptitle(f"{header.get('run_id')} gen {header.get('gen')} · {boss} "
                 f"· confidence trace", fontsize=10, color="#222")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def render_postmortem(pms, header, out: Path) -> None:
    """One strip per death: action-class band, boss-state band, khp/soul,
    V(s), over the last ~5 s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    actions = header["actions"]
    states = header["boss_spec"]["fsm_states"]
    state_color = {s: plt.get_cmap("tab20")(i % 20)
                   for i, s in enumerate(states)}
    n = len(pms)
    fig, axes = plt.subplots(3 * n, 1, figsize=(8.5, 3.4 * n), dpi=150,
                             squeeze=False,
                             gridspec_kw={"height_ratios": [1, 2, 2] * n})
    fig.patch.set_facecolor("white")
    for k, pm in enumerate(pms):
        steps = pm["steps"]
        t = [(s["i"] - steps[-1]["i"]) / 15 for s in steps]  # death at 0
        bands, hp, val = (axes[3 * k][0], axes[3 * k + 1][0],
                          axes[3 * k + 2][0])
        for x, s in zip(t, steps):
            bands.bar(x, 1, width=1 / 15, align="edge",
                      color=CLASS_COLORS[action_class(actions[s["a"]])])
            bands.bar(x, 1, bottom=1.1, width=1 / 15, align="edge",
                      color=state_color.get(s["obs"]["boss_state"], "#ccc"))
        bands.set_ylim(0, 2.1)
        bands.set_yticks([0.5, 1.6], ["action", "boss"], fontsize=7)
        killing = ", ".join(f"{k2} {v:+.1f}"
                            for k2, v in pm["killing_terms"].items())
        bands.set_title(f"episode {pm['ep']} death · step "
                        f"{pm['total_steps']} · {killing}",
                        fontsize=9, color="#222")
        hp.step(t, [s["obs"]["khp"] for s in steps], where="post",
                color="#e34948", label="khp")
        hp.plot(t, [s["obs"]["soul"] / 11 for s in steps], color="#2a78d6",
                lw=1.1, label="soul/11")
        hp.set_ylim(0, 9.5)
        hp.legend(fontsize=7, frameon=False)
        val.plot(t, [s["v"] for s in steps], color="#12314e", lw=1.4)
        val.set_ylabel("V(s)", fontsize=8)
        val.set_xlabel("seconds before death", fontsize=8)
        _mute_axes(bands, hp, val)
    fig.legend(handles=[Patch(color=c, label=l)
                        for l, c in CLASS_COLORS.items()],
               ncol=7, fontsize=7, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _add_trace(sub):
    ap = sub.add_parser("trace", help="per-episode confidence timeline")
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.set_defaults(run=_run_trace)


def _run_trace(args):
    recs = [read_recording(p.expanduser()) for p in args.recordings]
    header, _, _ = merge_recordings(recs)   # boss check + batch header only
    # Per-recording: episode numbers restart at 1 in every file, so each
    # file traces separately rather than over concatenated rows.
    trace = [ep for rec in recs for ep in confidence_trace(rec)]
    out = args.out or _default_out(args.recordings, ".trace.png")
    render_trace(trace, header, out)
    print(f"{len(trace)} episode traces -> {out}", flush=True)


def _add_postmortem(sub):
    ap = sub.add_parser("postmortem", help="last ~5 s before each death")
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--window", type=int, default=75)
    ap.set_defaults(run=_run_postmortem)


def _run_postmortem(args):
    recs = [read_recording(p.expanduser()) for p in args.recordings]
    header, _, _ = merge_recordings(recs)
    pms = [pm for rec in recs for pm in postmortems(rec, window=args.window)]
    if not pms:
        print("no losses in the given recordings; nothing to render",
              flush=True)
        return
    out = args.out or _default_out(args.recordings, ".postmortem.png")
    render_postmortem(pms, header, out)
    print(f"{len(pms)} deaths -> {out}", flush=True)


def render_reaction(prof, header, out: Path) -> None:
    """Distance-bucket x action-class heatmap: what comes out in the half
    second after a projectile appears at that range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    kept = [b for b in prof["buckets"] if b["n"]]
    classes = prof["classes"]
    m = np.array([[b["shares"][c] for c in classes] for b in kept])
    fig, ax = plt.subplots(figsize=(0.9 * len(classes) + 3.0,
                                    0.5 * len(kept) + 2.2), dpi=150)
    fig.patch.set_facecolor("white")
    im = ax.imshow(m, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, fontsize=8, color="#333")
    ax.set_yticks(range(len(kept)),
                  [f"{b['name']} ({b['lo']:g}–{b['hi']:g})  (n={b['n']})"
                   for b in kept], fontsize=8, color="#333")
    ax.set_xticks([x - 0.5 for x in range(1, len(classes))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(kept))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for r, row in enumerate(m):
        for c, v in enumerate(row):
            if v >= 0.1:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if v > 0.55 else "#1a3a5c")
    boss = header.get("boss_spec", {}).get("display_name", header.get("boss"))
    ax.set_title(f"{header.get('run_id')} gen {header.get('gen')} · {boss} "
                 f"· action mix in the {prof['window']} steps after a "
                 f"projectile appears · {prof['onsets']} onsets",
                 fontsize=9, color="#222", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("share of steps", fontsize=8, color="#333")
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)
    empty = [b["name"] for b in prof["buckets"] if not b["n"]]
    if empty:
        fig.text(0.01, 0.01, f"no onsets in: {', '.join(empty)}",
                 fontsize=7, color="#777")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def render_soul(econ, header, out: Path) -> None:
    """Two side-by-side HP x SOUL heatmaps: P(cast) and P(focus), with the
    per-cell step count printed so an empty region reads as no-data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    buckets = econ["soul_buckets"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.6), dpi=150)
    fig.patch.set_facecolor("white")
    for ax, key in zip(axes, ("cast", "focus")):
        # khp 9 at the top: reverse the row order for display.
        m = np.array([[cell[key] for cell in row]
                      for row in reversed(econ["cells"])])
        ns = [[cell["n"] for cell in row] for row in reversed(econ["cells"])]
        im = ax.imshow(m, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(buckets)), buckets, fontsize=8, color="#333")
        ax.set_yticks(range(9), [str(h) for h in reversed(econ["khp"])],
                      fontsize=8, color="#333")
        ax.set_xlabel("SOUL", fontsize=8)
        ax.set_ylabel("masks (khp)", fontsize=8)
        ax.set_xticks([x - 0.5 for x in range(1, len(buckets))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, 9)], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for r in range(9):
            for c in range(len(buckets)):
                if ns[r][c]:
                    ax.text(c, r, f"{m[r][c]:.2f}\nn={ns[r][c]}",
                            ha="center", va="center", fontsize=5.5,
                            color="white" if m[r][c] > 0.55 else "#1a3a5c")
        ax.set_title(f"P({key} | HP, SOUL)", fontsize=9, color="#222")
    boss = header.get("boss_spec", {}).get("display_name", header.get("boss"))
    fig.suptitle(f"{header.get('run_id')} gen {header.get('gen')} · {boss} "
                 f"· SOUL economy", fontsize=10, color="#222")
    fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02,
                 label="probability").outline.set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def render_actionmix(mix, header, out: Path) -> None:
    """Class share vs generation, one line per class that ever exceeds 1%."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    for cls in mix["classes"]:
        ys = [r["shares"][cls] for r in mix["rows"]]
        if max(ys) <= 0.01:
            continue                      # a never-used class earns no line
        ax.plot(mix["gens"], ys, color=CLASS_COLORS[cls], lw=2,
                marker="o", ms=4, label=cls)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("generation", fontsize=8)
    ax.set_ylabel("share of steps", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=4)
    ax.tick_params(labelsize=7, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(color="#eee", lw=0.6)
    boss = header.get("boss_spec", {}).get("display_name", header.get("boss"))
    episodes = sum(r["episodes"] for r in mix["rows"])
    ax.set_title(f"{header.get('run_id')} · {boss} · action mix across "
                 f"{len(mix['gens'])} recorded generations "
                 f"({episodes} episodes)", fontsize=10, color="#222",
                 pad=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _add_reaction(sub):
    ap = sub.add_parser("reaction", help="action mix after projectile onset")
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--window", type=int, default=8)
    ap.set_defaults(run=_run_reaction)


def _run_reaction(args):
    recs = [read_recording(p.expanduser()) for p in args.recordings]
    header, _, _ = merge_recordings(recs)
    # Episode numbers restart at 1 in every file; renumber while
    # concatenating so onset detection never sees a false flip across a
    # file boundary.
    combined, offset = [header], 0
    for rec in recs:
        max_ep = 0
        for r in rec:
            if r["type"] == "step":
                combined.append({**r, "ep": r["ep"] + offset})
                max_ep = max(max_ep, r["ep"])
        offset += max_ep
    prof = reaction_profile(combined, window=args.window)
    if prof["onsets"] == 0:
        print("no projectile onsets in the given recordings; nothing to "
              "render", flush=True)
        return
    out = args.out or _default_out(args.recordings, ".reaction.png")
    render_reaction(prof, header, out)
    print(f"{prof['onsets']} onsets -> {out}", flush=True)


def _add_soul(sub):
    ap = sub.add_parser("soul", help="heal-vs-cast over the HP x SOUL grid")
    ap.add_argument("recordings", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.set_defaults(run=_run_soul)


def _run_soul(args):
    recs = [read_recording(p.expanduser()) for p in args.recordings]
    header, steps, _ = merge_recordings(recs)
    econ = soul_economy([header] + steps)
    out = args.out or _default_out(args.recordings, ".soul.png")
    render_soul(econ, header, out)
    print(f"{len(steps)} steps -> {out}", flush=True)


def _add_actionmix(sub):
    ap = sub.add_parser("actionmix",
                        help="class shares across recorded generations")
    ap.add_argument("recordings", nargs="+", type=Path,
                    help="recordings of one run, any mix of generations")
    ap.add_argument("--out", type=Path, default=None)
    ap.set_defaults(run=_run_actionmix)


def _run_actionmix(args):
    recs = [read_recording(p.expanduser()) for p in args.recordings]
    mix = action_mix(recs)
    out = args.out or _default_out(args.recordings, ".action_mix.png")
    render_actionmix(mix, recs[0][0], out)
    print(f"{len(mix['gens'])} generations -> {out}", flush=True)


def _default_out(recordings, suffix: str) -> Path:
    first = recordings[0].expanduser()
    return first.parent / (first.name.replace(".jsonl.gz", "") + suffix)


def _add_matrix(sub):
    ap = sub.add_parser("matrix", help="action x boss-state heatmap")
    ap.add_argument("recordings", nargs="+", type=Path,
                    help="schema-v1 .jsonl.gz recording file(s), one boss")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <first recording>"
                         ".action_matrix.png)")
    ap.add_argument("--min-steps", type=int, default=5,
                    help="drop states observed fewer than this many times")
    ap.set_defaults(run=_run_matrix)


def _run_matrix(args):
    recs = [read_recording(p.expanduser()) for p in args.recordings]
    header, steps, episodes = merge_recordings(recs)
    agg = aggregate(steps, min_steps=args.min_steps)
    out = args.out or _default_out(args.recordings, ".action_matrix.png")
    render(agg, header, episodes, out)
    print(f"{len(agg.states)} states x {len(header['actions'])} actions "
          f"from {sum(agg.counts)} steps -> {out}", flush=True)


SUBCOMMANDS = ("matrix", "trace", "postmortem", "reaction", "soul",
               "actionmix")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Back-compat shim: `analyze_steps.py rec.jsonl.gz` predates the
    # subcommands and keeps meaning `matrix rec.jsonl.gz`.
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "matrix")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(required=True)
    _add_matrix(sub)
    _add_trace(sub)
    _add_postmortem(sub)
    _add_reaction(sub)
    _add_soul(sub)
    _add_actionmix(sub)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
