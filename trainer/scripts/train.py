#!/usr/bin/env python3
"""Train PPO against Hornet 1 on one supervised game instance.

Owns the game process end to end: launches it, supervises it through
crashes, wedges, and App Nap suspensions, checkpoints a generation every
--gen-every timesteps, and shuts it down on exit. Stop a run with one
Ctrl-C: it finishes the current rollout, saves a final generation, and
terminates the game. launch_instances.py stays for manual gates only -- the
training game must be this process's own child or relaunch() could never
reap a wedged game's port.
"""
import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    VecFrameStack, VecMonitor, VecNormalize,
)

from hkrl.game import GameProcess  # noqa: E402
from hkrl.generations import GenerationCallback, latest_checkpoint  # noqa: E402
from hkrl.supervisor import InstanceDown, SupervisedVecEnv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_instances import DEFAULT_APP, DEFAULT_PORT  # noqa: E402

GAMMA = 0.995
N_STACK = 4


def build_env(ports, relaunch, run_dir, **supervisor_kwargs):
    """SupervisedVecEnv -> VecMonitor -> VecFrameStack -> VecNormalize.

    Returns (env, supervisor): the outermost wrapper for PPO, plus the
    supervisor itself so the checkpoint callback can read its recovery
    count.

    VecMonitor sits below the frame stack and the normalizer so its episode
    records carry raw rewards and true lengths. It gets no info_keywords:
    VecMonitor indexes every keyword into each done step's info, and the
    supervisor's recovery frames carry only terminal_observation, so a
    keyword would KeyError there and kill the run on its first recovery.
    GenerationCallback reads won/boss_damage_frac from the raw infos
    instead.

    VecFrameStack(n_stack=4): one observation is an instant, and the FSM
    one-hot does not encode how long Hornet has been in a state -- the
    stack is what lets the policy tell "starting a dash" from "mid-dash".

    VecNormalize shares PPO's gamma so its return normalization tracks the
    same discounted quantity the value head learns.
    """
    supervisor = SupervisedVecEnv(list(ports), relaunch=relaunch, **supervisor_kwargs)
    session = time.strftime("%Y%m%d_%H%M%S")
    # Session-stamped so a resumed run appends a new episode log instead of
    # truncating the previous session's.
    mon = VecMonitor(supervisor, filename=str(Path(run_dir) / f"monitor_{session}"))
    stacked = VecFrameStack(mon, n_stack=N_STACK)
    env = VecNormalize(stacked, gamma=GAMMA, clip_obs=10.0)
    return env, supervisor


def build_model(env, run_dir, seed=None, n_steps=2048, batch_size=64, n_epochs=10):
    """A fresh PPO for this env."""
    return PPO(
        "MlpPolicy",
        env,
        # ~13s credit horizon at 15 Hz, so a dodge can still be credited
        # with the punish it enables; the default 0.99 covers only ~6.7s.
        gamma=GAMMA,
        learning_rate=3e-4,
        # One instance at 15 Hz collects the full n_steps=2048 (~2.3 minutes
        # of play) per update.
        n_steps=n_steps,
        batch_size=batch_size,
        # The whole gradient update runs while the game connection idles,
        # and the mod drops any connection idle for 10s (see the ceiling
        # note in hkrl/env.py) -- so the update must finish well inside 10s
        # or every rollout boundary severs the connection. Measured at the
        # live gate; if updates approach 10s, lower --n-epochs first.
        n_epochs=n_epochs,
        # Discrete 15-action combat collapses quickly into "hold nothing"
        # under a death-dominated reward; a small entropy bonus keeps
        # alternatives alive long enough for the damage term to be
        # discovered.
        ent_coef=0.01,
        seed=seed,
        verbose=1,
        # The policy is a tiny MLP; CPU avoids the per-batch device
        # transfer overhead that would dominate on MPS.
        device="cpu",
        tensorboard_log=str(Path(run_dir) / "tb"),
    )


class StopOnFlag(BaseCallback):
    """Ends learn() cleanly when the flag is set: rollout collection stops at
    the next step boundary and learn() returns, so the final checkpoint save
    and game shutdown run on the normal path instead of unwinding through a
    KeyboardInterrupt raised inside a socket read or a recovery."""

    def __init__(self, flag: threading.Event):
        super().__init__()
        self.flag = flag

    def _on_step(self) -> bool:
        return not self.flag.is_set()
