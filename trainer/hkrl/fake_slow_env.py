"""In-process fakes for async-reset tests: envs whose reset() takes time.

Lives in hkrl (like fake_game.py) rather than tests/ so SubprocVecEnv's
spawned workers can import the classes -- test modules aren't importable
from a spawned child.
"""
import threading
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hkrl.async_reset import AsyncResetWrapper


class SlowResetEnv(gym.Env):
    """Toy env standing in for HKEnv, with a test-controlled blocking reset.

    reset() parks on `gate` until the test releases it (the gate starts
    open, so undisturbed resets complete immediately). Observations are the
    reset count broadcast over the space, so a test can tell exactly which
    reset produced the frame it is looking at. Every action that actually
    reaches this env lands in `actions` -- the no-action-while-pending
    invariant is asserted against that list.
    """

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.gate = threading.Event()
        self.gate.set()
        self.aborted = threading.Event()
        self.resets = 0
        self.actions = []
        self.next_done = False    # set by tests: the next step ends its episode
        self.reset_error = None   # set by tests: reset raises this instead

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gate.wait()
        if self.aborted.is_set():
            raise RuntimeError("reset aborted")
        if self.reset_error is not None:
            raise self.reset_error
        self.resets += 1
        return self._obs(), {"reset": self.resets}

    def step(self, action):
        self.actions.append(int(action))
        terminated, self.next_done = self.next_done, False
        return self._obs(), 1.0, terminated, False, {}

    def abort_reset(self):
        self.aborted.set()
        self.gate.set()

    def _obs(self):
        return np.full(3, float(self.resets), dtype=np.float32)


class TimedResetEnv(gym.Env):
    """Picklable toy for SubprocVecEnv tests: reset() sleeps `reset_s`,
    episodes end after `episode_len` steps. Event-gated SlowResetEnv can't
    cross a process boundary; wall-clock sleeps can."""

    def __init__(self, reset_s=0.0, episode_len=3):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.reset_s = reset_s
        self.episode_len = episode_len
        self.resets = 0
        self._t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(self.reset_s)
        self.resets += 1
        self._t = 0
        return np.full(3, float(self.resets), dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.episode_len
        return (np.full(3, float(self.resets), dtype=np.float32),
                1.0, terminated, False, {})


def make_async_timed(reset_s, episode_len, pending_mode="prefix"):
    """Module-level factory: picklable for SubprocVecEnv spawn workers."""
    return AsyncResetWrapper(TimedResetEnv(reset_s, episode_len),
                             pending_mode=pending_mode,
                             placeholder_tick_s=0.0)
