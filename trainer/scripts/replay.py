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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv, VecNormalize,
)

from hkrl.generations import checkpoint_paths, latest_checkpoint  # noqa: E402
from hkrl.vec import make_env  # noqa: E402


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


def replay(model, env, episodes: int, out=None, deterministic: bool = True):
    """Run episodes and print one summary line per episode; returns the
    summary dicts. Relies on the vec env's autoreset: the observation
    returned by a done step is already the next episode's first."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--gen", type=int, default=None,
                    help="generation number (default: the run's latest)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9020)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy instead of argmax actions")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser()
    if args.gen is None:
        gen, weights, vecnorm = latest_checkpoint(run_dir)
    else:
        gen = args.gen
        weights, vecnorm = checkpoint_paths(run_dir, gen)
    print(f"replaying generation {gen} from {run_dir}", flush=True)

    model, env = load_policy(weights, vecnorm, port=args.port, host=args.host)
    try:
        summaries = replay(model, env, episodes=args.episodes,
                           deterministic=not args.stochastic)
    finally:
        env.close()
    wins = sum(1 for s in summaries if s["won"])
    damage = sum(s["boss_damage_frac"] for s in summaries) / len(summaries)
    print(f"\n{wins}/{len(summaries)} won, mean boss damage "
          f"{damage * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
