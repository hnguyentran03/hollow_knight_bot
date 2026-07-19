"""Vectorized environment over several game instances."""
from typing import Callable, Sequence

from stable_baselines3.common.vec_env import SubprocVecEnv

from hkrl.env import HKEnv


def make_env(port: int, **env_kwargs) -> Callable[[], HKEnv]:
    """Return a factory that builds an HKEnv bound to one instance.

    A factory rather than an instance: HKEnv connects in __init__, so
    recovering a crashed slot means constructing a fresh env, not reusing
    a dead one.
    """
    def _init() -> HKEnv:
        return HKEnv(port=port, **env_kwargs)
    return _init


def make_vec(ports: Sequence[int], **env_kwargs) -> SubprocVecEnv:
    """Build a SubprocVecEnv, one subprocess per instance.

    Must be SubprocVecEnv, not DummyVecEnv: the wire protocol is lockstep
    and blocking, so a single-process vec env would serialize the socket
    waits and deliver no speedup at all.
    """
    return SubprocVecEnv([make_env(p, **env_kwargs) for p in ports])
