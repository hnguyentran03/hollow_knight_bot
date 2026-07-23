"""Vectorized environment over one game instance per port.

Training runs a single instance today, so `ports` is normally one long; PPO
needs a VecEnv either way.
"""
import time
from typing import Callable, Sequence

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from hkrl.async_reset import AsyncResetWrapper
from hkrl.env import HKEnv


class RealEpisodeVecMonitor(VecMonitor):
    """VecMonitor that drops async-reset throwaway episodes.

    AsyncResetWrapper's "isolated" mode ends each pending window as its own
    zero-reward episode whose done step carries reset_pending=True in info.
    Stock VecMonitor records those like real fights, polluting the monitor
    CSV (and through it the dashboard) plus rollout/ep_rew_mean, so this
    subclass zeroes the accumulators without writing a row or attaching
    info["episode"]. Reimplements step_wait rather than delegating: the
    parent writes its CSV row inside the loop, too late to take back.
    """

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if not dones[i]:
                continue
            if infos[i].get("reset_pending"):
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                continue
            info = infos[i].copy()
            episode_info = {"r": self.episode_returns[i],
                            "l": self.episode_lengths[i],
                            "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                episode_info[key] = info[key]
            info["episode"] = episode_info
            self.episode_count += 1
            self.episode_returns[i] = 0
            self.episode_lengths[i] = 0
            if self.results_writer:
                self.results_writer.write_row(episode_info)
            new_infos[i] = info
        return obs, rewards, dones, new_infos


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
