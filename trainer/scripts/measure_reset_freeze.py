#!/usr/bin/env python3
"""Phase 0 async-resets report: how much fleet wall-clock a run lost to
sibling freeze.

Reads the resets_<port>.jsonl sidecars a `train.py --measure-resets` run
wrote, resolves the fleet size (config.jsonl) and rollout wall-clock (the last
generation in generations.jsonl), and prints the freeze fraction -- the
go/no-go signal for the async-reset work (see
docs/superpowers/specs/2026-07-21-async-resets-design.md, Phase 0).

    ./.venv/bin/python scripts/measure_reset_freeze.py ~/hkrl/runs/<run_id>

Override N or wall-clock when the run files can't supply them (e.g. a run
stopped before its first checkpoint):

    ... <run_dir> --instances 2 --wallclock-s 1800
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkrl.reset_metrics import report_run  # noqa: E402


def format_report(s: dict) -> str:
    pct = s["exclusive_freeze_fraction"] * 100.0
    naive = s["freeze_fraction"] * 100.0
    # A directional read of the go/no-go signal, not a verdict: the design's
    # gate is a training-quality comparison, not a wall-clock threshold.
    read = ("substantial -- worth building async resets"
            if pct >= 10.0 else
            "small -- may not be worth the async correctness risk")
    return "\n".join([
        f"run instances:      {s['n_instances']}",
        f"rollout wall-clock:  {s['wallclock_s']:.1f}s",
        f"resets measured:     {s['n_resets']}",
        f"reset span total:    {s['total_reset_s']:.1f}s",
        f"reset span mean/max: {s['mean_reset_s']:.1f}s / {s['max_reset_s']:.1f}s",
        f"sibling-freeze:      {pct:.1f}% of fleet wall-clock reclaimable "
        f"(overlap-aware; naive upper bound {naive:.1f}%)",
        f"  ^ {read}.",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a run directory")
    ap.add_argument("--instances", type=int, default=None,
                    help="override the fleet size (default: read config.jsonl)")
    ap.add_argument("--wallclock-s", type=float, default=None,
                    help="override rollout wall-clock seconds "
                         "(default: last generation's wall_clock_s)")
    args = ap.parse_args()
    if not args.run_dir.is_dir():
        sys.exit(f"{args.run_dir} is not a directory")
    s = report_run(args.run_dir, n_instances=args.instances,
                   wallclock_s=args.wallclock_s)
    if s["n_resets"] == 0:
        sys.exit(f"no reset spans under {args.run_dir} -- was the run started "
                 f"with --measure-resets?")
    if s["wallclock_s"] <= 0:
        # The freeze fraction divides by rollout wall-clock; with none known
        # (no completed episode and no checkpoint yet) it would read a
        # meaningless 0%. Report the raw spans and ask for the wall-clock
        # rather than print a number that looks like "no freeze".
        sys.exit(
            f"{s['n_resets']} resets totaling {s['total_reset_s']:.1f}s "
            f"measured, but the run's rollout wall-clock is unknown (no "
            f"completed episode or checkpoint yet). Re-run with "
            f"--wallclock-s <seconds the run actually ran> to get the "
            f"sibling-freeze fraction.")
    print(format_report(s))


if __name__ == "__main__":
    main()
