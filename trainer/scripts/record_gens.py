#!/usr/bin/env python3
"""Batch-record historical checkpoints: one game, many generations.

Fuel for the action-mix-across-generations view: replays every selected
checkpoint with --record semantics into <run-dir>/replays/, one standard
recording file per generation, against a single launched game (the mod's
AcceptLoop adopts each fresh connection, so switching checkpoints is just
a reconnect -- no relaunch). Ctrl-C stops at the next episode boundary;
completed recordings are kept.
"""
import argparse
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hkrl.generations import checkpoint_paths, last_generation  # noqa: E402
from hkrl.recording import (RecordingWriter, build_header,  # noqa: E402
                            recording_path)
from hkrl.rundata import run_boss  # noqa: E402
from launch_instances import DEFAULT_APP, DEFAULT_PORT  # noqa: E402
from replay import auto_game, load_policy, record_replay  # noqa: E402


def select_gens(run_dir, every=None, gens=None) -> list[int]:
    """The generations to record, ascending. --every N strides the run's
    complete checkpoints from the oldest and always includes the newest
    (the current bot belongs in any evolution picture); --gens is explicit
    and errors on a missing checkpoint rather than guessing."""
    run_dir = Path(run_dir)
    available = [g for g in range(1, last_generation(run_dir) + 1)
                 if all(p.exists() for p in checkpoint_paths(run_dir, g))]
    if gens is not None:
        missing = sorted(set(gens) - set(available))
        if missing:
            raise FileNotFoundError(
                f"no complete checkpoint for generation(s) "
                f"{', '.join(map(str, missing))} under {run_dir}")
        return sorted(set(gens))
    picked = available[::every]
    if available and available[-1] not in picked:
        picked.append(available[-1])
    return picked


def _stopper(stop):
    def handler(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt  # second Ctrl-C: abandon the clean path
        stop.set()
        print("stop requested: finishing the current episode; completed "
              "recordings are kept (Ctrl-C again to force)",
              file=sys.stderr, flush=True)
    return handler


def record_generations(run_dir, gens, *, root, app=DEFAULT_APP,
                       port=DEFAULT_PORT, episodes=3, headless=False,
                       stop=None) -> list[Path]:
    """Record each generation into its own <run-dir>/replays/ file against
    one shared game. A set `stop` skips the remaining generations after
    the in-flight episode finishes (record_replay honors it per episode)."""
    run_dir = Path(run_dir)
    written = []
    with auto_game(root, app, port, headless=headless) as game:
        if stop is not None:
            # After start() so a Ctrl-C during the cold boot unwinds
            # through auto_game's reap rather than this handler.
            signal.signal(signal.SIGINT, _stopper(stop))
        for gen in gens:
            if stop is not None and stop.is_set():
                break
            weights, vecnorm = checkpoint_paths(run_dir, gen)
            out = recording_path(run_dir / "replays", gen)
            print(f"--- generation {gen} -> {out.name}", flush=True)
            model, env = load_policy(weights, vecnorm, port=game.ports[0],
                                     run_dir=run_dir, capture=True)
            try:
                with RecordingWriter(out) as writer:
                    writer.header(**build_header(
                        run_dir=run_dir, gen=gen, weights=weights,
                        vecnorm=vecnorm, boss_id=run_boss(run_dir),
                        deterministic=True, episodes=episodes,
                        headless=headless, auto=True))
                    record_replay(model, env, episodes=episodes,
                                  writer=writer, stop=stop)
            finally:
                env.close()
            written.append(out)
    return written


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    pick = ap.add_mutually_exclusive_group(required=True)
    pick.add_argument("--every", type=int,
                      help="record every Nth complete checkpoint "
                           "(plus the newest)")
    pick.add_argument("--gens",
                      type=lambda s: [int(g) for g in s.split(",")],
                      help="explicit comma-separated generation list")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--root", type=Path, default=Path("~/hkrl").expanduser(),
                    help="save-backup / instance-clone root (mirrors "
                         "replay.py --auto)")
    ap.add_argument("--headless", action="store_true",
                    help="launch the game -batchmode -nographics")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.expanduser()
    gens = select_gens(run_dir, every=args.every, gens=args.gens)
    if not gens:
        sys.exit(f"no complete checkpoints under {run_dir}")
    print(f"recording {len(gens)} generations ({gens[0]}..{gens[-1]}) x "
          f"{args.episodes} episodes -- roughly "
          f"{max(1, len(gens) * args.episodes * 30 // 60)} min of game "
          f"time", flush=True)
    written = record_generations(
        run_dir, gens, root=args.root.expanduser(), port=args.port,
        episodes=args.episodes, headless=args.headless,
        stop=threading.Event())
    print(f"{len(written)} recordings written to {run_dir / 'replays'}",
          flush=True)


if __name__ == "__main__":
    main()
