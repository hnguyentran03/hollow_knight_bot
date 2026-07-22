"""Vectorized environment over one game instance per port.

Training runs a single instance today, so `ports` is normally one long; PPO
needs a VecEnv either way.
"""
from typing import Callable, Sequence

from stable_baselines3.common.vec_env import SubprocVecEnv

from hkrl.async_reset import AsyncResetWrapper
from hkrl.env import HKEnv


def make_env(port: int, async_resets: bool = False,
             pending_mode: str = "prefix", **env_kwargs) -> Callable[[], HKEnv]:
    """Return a factory that builds an HKEnv bound to one instance.

    A factory rather than an instance: HKEnv connects in __init__, so
    recovering a crashed slot means constructing a fresh env, not reusing
    a dead one. `async_resets` wraps the env in AsyncResetWrapper (see
    hkrl/async_reset.py) -- a multi-instance throughput feature, so N=1
    callers leave it off; remaining kwargs pass to HKEnv untouched.
    """
    def _init() -> HKEnv:
        env = HKEnv(port=port, **env_kwargs)
        if async_resets:
            return AsyncResetWrapper(env, pending_mode=pending_mode)
        return env
    return _init


def make_vec(ports: Sequence[int], **env_kwargs) -> SubprocVecEnv:
    """Build a SubprocVecEnv, one subprocess per instance.

    Must be SubprocVecEnv, not DummyVecEnv: the wire protocol is lockstep
    and blocking, so a single-process vec env would serialize the socket
    waits and deliver no speedup at all.
    """
    return SubprocVecEnv([make_env(p, **env_kwargs) for p in ports])
