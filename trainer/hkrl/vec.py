"""Vectorized environment over one game instance per port.

Training runs a single instance today, so `ports` is normally one long; PPO
needs a VecEnv either way.
"""
import time
from typing import Callable, Sequence

import numpy as np

from stable_baselines3.common.vec_env import (SubprocVecEnv, VecMonitor,
                                              VecNormalize)

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
            # Tolerate absent keywords rather than index them: the
            # supervisor's recovery frames carry only terminal_observation,
            # so stock VecMonitor's info[key] would KeyError there and kill
            # the run on its first recovery. The CSV writer fills the blank.
            for key in self.info_keywords:
                if key in info:
                    episode_info[key] = info[key]
            info["episode"] = episode_info
            self.episode_count += 1
            self.episode_returns[i] = 0
            self.episode_lengths[i] = 0
            if self.results_writer:
                self.results_writer.write_row(episode_info)
            new_infos[i] = info
        return obs, rewards, dones, new_infos


class RealEpisodeVecNormalize(VecNormalize):
    """VecNormalize whose running statistics learn only from real frames.

    Async-reset placeholder steps (info reset_pending=True, all-zero
    observations, zero reward -- hkrl/async_reset.py) are ~19% of what a
    two-instance isolated run collects. Stock VecNormalize folds them into
    obs_rms and ret_rms, dragging the observation mean toward the zero
    placeholder and the return variance toward streams of zero reward, so
    every REAL frame gets systematically mis-normalized. This subclass
    updates the statistics from real rows only; normalization itself is
    still applied to every row, since the policy still receives placeholder
    observations. Reimplements step_wait rather than delegating: the parent
    updates its statistics inline, too early to take back.

    (The autoreset frame a real episode ends on is a placeholder without
    the flag, and the one a throwaway window ends on is real WITH the flag;
    the two miscounts are rare -- one each per reset -- and cancel.)
    """

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        assert isinstance(obs, np.ndarray)  # HK observations are a flat Box
        self.old_obs = obs
        self.old_reward = rewards
        real = np.array([not info.get("reset_pending") for info in infos],
                        dtype=bool)

        if self.training:
            if self.norm_obs and real.any():
                self.obs_rms.update(obs[real])
            self.returns = self.returns * self.gamma + rewards
            if real.any():
                self.ret_rms.update(self.returns[real])

        obs = self.normalize_obs(obs)
        rewards = self.normalize_reward(rewards)

        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(
                    infos[idx]["terminal_observation"])

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    @classmethod
    def load(cls, load_path, venv):
        vec = VecNormalize.load(load_path, venv)
        # Pre-guard checkpoints unpickle as plain VecNormalize; the guard
        # is pure behavior (no extra state), so upgrading the class is
        # enough to keep it across resumes.
        vec.__class__ = cls
        return vec


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
