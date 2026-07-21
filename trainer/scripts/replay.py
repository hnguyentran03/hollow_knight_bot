#!/usr/bin/env python3
"""Replay a saved generation against a live game instance.

Watching an early and a late generation fight the same boss is the point of
checkpointing them separately. Loads gen_NNNN.zip together with its
gen_NNNN_vecnorm.pkl -- the weights are meaningless under any other
observation statistics -- freezes the statistics (training=False), and
reports real rewards (norm_reward=False).

Never point --port at the port a live training run is using: the mod's
AcceptLoop keeps only its newest client, so connecting here would sever the
trainer's connection and force a recovery. Replay after training stops,
against an instance launched with scripts/launch_instances.py -- with HOME
isolation gone, there is no isolated second instance to replay against
while training runs.
"""
import argparse
import signal
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv, VecNormalize,
)

from hkrl.game import GameFleet  # noqa: E402
from hkrl.generations import checkpoint_paths, latest_checkpoint  # noqa: E402
from hkrl.vec import make_env  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_instances import (  # noqa: E402
    DEFAULT_APP, DEFAULT_PORT, SAVE_ISOLATION_SUPPORTED, backup_saves,
    prepare_instance,
)


def load_policy(weights: Path, vecnorm: Path, port: int, host: str = "127.0.0.1"):
    """The training pipeline minus training: one env, normalized.

    Mirrors scripts/train.py's build_env exactly (SupervisedVecEnv there,
    DummyVecEnv here): no VecFrameStack, because the recurrent MlpLstmPolicy
    carries its own temporal memory and the checkpoint's weights only make
    sense against the same observation pipeline they were trained under.

    DummyVecEnv rather than Subproc: with a single env there are no parallel
    socket waits to overlap, so a subprocess would add IPC for nothing.
    """
    venv = DummyVecEnv([make_env(port, host=host)])
    env = VecNormalize.load(str(vecnorm), venv)
    env.training = False     # statistics are a checkpoint artifact, frozen here
    env.norm_reward = False  # report the env's real rewards, not scaled ones
    model = RecurrentPPO.load(str(weights), device="cpu")
    return model, env


def replay(model, env, episodes: int, out=None, deterministic: bool = True,
           stop: threading.Event | None = None):
    """Run episodes and print one summary line per episode; returns the
    summary dicts. Relies on the vec env's autoreset: the observation
    returned by a done step is already the next episode's first.

    `stop`, when given, ends the loop at the next episode boundary -- the
    same episode-boundary semantics train.py's StopOnFlag uses, so the
    dashboard's single Stop (one SIGINT sets the flag) lets the fight in
    progress finish instead of severing it mid-swing."""
    if out is None:
        out = sys.stdout
    summaries = []
    obs = env.reset()
    # RecurrentPPO threads an LSTM hidden state between steps: predict()
    # takes the previous state and an episode_start mask (which zeroes the
    # state at an episode boundary) and returns the updated state. Feeding
    # None + all-True on the first call initializes it, and setting
    # episode_starts from `dones` resets the memory exactly when the vec env
    # autoresets, so each episode's policy starts from a clean hidden state
    # instead of carrying the previous fight's memory into the new one.
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    for ep in range(1, episodes + 1):
        if stop is not None and stop.is_set():
            break
        total, steps, done, infos = 0.0, 0, False, [{}]
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts,
                deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            episode_starts = dones
            total += float(rewards[0])
            steps += 1
            done = bool(dones[0])
        info = infos[0]
        won = bool(info.get("won", False))
        truncated = bool(info.get("TimeLimit.truncated", False))
        result = "WIN" if won else ("TIMEOUT" if truncated else "loss")
        summary = dict(episode=ep, result=result, steps=steps, reward=total,
                       won=won,
                       boss_damage_frac=float(info.get("boss_damage_frac", 0.0)),
                       attempt=info.get("attempt"))
        summaries.append(summary)
        print(f"episode {ep:3d} (attempt {info.get('attempt')}): "
              f"result={result:7s} steps={steps:4d} reward={total:8.2f} "
              f"boss_dmg={summary['boss_damage_frac'] * 100:5.1f}%",
              file=out, flush=True)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--gen", type=int, default=None,
                    help="generation number (default: the run's latest)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--root", type=Path, default=Path("~/hkrl").expanduser(),
                    help="save-backup / instance-clone root for --auto "
                         "(mirrors train.py's --root)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy instead of argmax actions")
    ap.add_argument("--auto", action="store_true",
                    help="launch a game, replay against it, then shut it "
                         "down (dashboard-driven; no already-running game "
                         "needed and no human ready prompt)")
    return ap


def banner(gen, run_dir, episodes) -> str:
    """The dashboard tails this log; a one-line banner keeps it legible."""
    return (f"replaying generation {gen} from {run_dir} "
            f"({episodes} episodes)")


def _print_summary(summaries) -> None:
    wins = sum(1 for s in summaries if s["won"])
    damage = (sum(s["boss_damage_frac"] for s in summaries) / len(summaries)
              if summaries else 0.0)
    print(f"\n{wins}/{len(summaries)} won, mean boss damage "
          f"{damage * 100:.1f}%", flush=True)


def run_connected(weights, vecnorm, *, host, port, episodes, deterministic):
    """Replay against a game already running on --port (the default mode --
    behavior unchanged)."""
    model, env = load_policy(weights, vecnorm, port=port, host=host)
    try:
        return replay(model, env, episodes=episodes,
                      deterministic=deterministic)
    finally:
        env.close()


def run_auto(weights, vecnorm, *, root, app, port, episodes, deterministic):
    """Self-contained replay: launch one game, replay against it, shut it
    down. Mirrors train.py's game handling so the dashboard can drive a
    replay exactly as it drives a run.

    Save safety is train.py's, doubled: backup_saves(root) snapshots the
    master save first, and -- where APFS clones are supported -- the game
    plays an isolated slot-0 clone (prepare_instance), so its autosaves can
    never reach the master slot at all. Slot 0 also gives the clone a
    regular 1280x720 windowed size (seed_prefs), which is what makes a
    replay watchable rather than one of training's shrunken hidden windows.
    Where clones are unsupported the replay falls back to playing the master
    save directly (the historical N=1 path), backup-protected only.

    The boot macro, driven by the env's first reset(), walks the game into
    the Hall of Gods -- there is no human ready prompt under --auto. A single
    SIGINT ends the loop at the next episode boundary; a second forces an
    abort, and the finally always reaps the game."""
    backup = backup_saves(root)
    if backup is not None:
        print(f"master save backed up to {backup}", flush=True)
    apps = None
    if SAVE_ISOLATION_SUPPORTED:
        print(f"preparing an isolated game copy under {root / 'instances'} "
              f"(APFS clone; instant, near-zero disk)", flush=True)
        apps = [prepare_instance(port, app, root / "instances", slot=0)]
    else:
        print("WARNING: save isolation is not implemented on this platform; "
              "the replay plays the master save directly (backup taken "
              "above) at the game's default window size.", file=sys.stderr,
              flush=True)
    game = GameFleet([port], app=app, apps=apps)
    env = None
    stop = threading.Event()

    def request_stop(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt  # second Ctrl-C: abandon the clean path
        stop.set()
        print("stop requested: finishing the current episode, then shutting "
              "the game down (Ctrl-C again to force)", file=sys.stderr,
              flush=True)

    try:
        game.start()
        print(f"game up on port {game.ports[0]}", flush=True)
        # After start() so a Ctrl-C during the cold boot still unwinds
        # through the finally's game.stop() rather than this handler.
        signal.signal(signal.SIGINT, request_stop)
        model, env = load_policy(weights, vecnorm, port=port)
        return replay(model, env, episodes=episodes,
                      deterministic=deterministic, stop=stop)
    finally:
        if env is not None:
            env.close()
        game.stop()


def main() -> None:
    args = build_parser().parse_args()

    run_dir = args.run_dir.expanduser()
    if args.gen is None:
        gen, weights, vecnorm = latest_checkpoint(run_dir)
    else:
        gen = args.gen
        weights, vecnorm = checkpoint_paths(run_dir, gen)
    print(banner(gen, run_dir, args.episodes), flush=True)

    if args.auto:
        summaries = run_auto(
            weights, vecnorm, root=args.root.expanduser(), app=DEFAULT_APP,
            port=args.port, episodes=args.episodes,
            deterministic=not args.stochastic)
    else:
        summaries = run_connected(
            weights, vecnorm, host=args.host, port=args.port,
            episodes=args.episodes, deterministic=not args.stochastic)
    _print_summary(summaries)


if __name__ == "__main__":
    main()
