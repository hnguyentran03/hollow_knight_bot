import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from hkrl.async_reset import AsyncResetWrapper
from hkrl.env import HKEnv
from hkrl.fake_game import FakeGame, obs, state
from hkrl.vec import RealEpisodeVecMonitor, make_env, make_vec


def _episode(steps=3):
    frames = [state(obs()) for _ in range(steps)]
    frames.append(state(obs(), done=True))
    return frames


def test_make_vec_drives_every_instance():
    with FakeGame([_episode()]) as a, FakeGame([_episode()]) as b:
        vec = make_vec([a.port, b.port])
        try:
            obs = vec.reset()
            assert obs.shape[0] == 2
            obs, rewards, dones, infos = vec.step([0, 0])
            assert obs.shape[0] == 2
            assert len(rewards) == 2
        finally:
            vec.close()


class _ScriptedEnv(gym.Env):
    """Replays a fixed (reward, done, info) script, one entry per step."""

    observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,),
                                   dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, script):
        self._script = list(script)

    def reset(self, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        reward, done, info = self._script.pop(0)
        return np.zeros(2, dtype=np.float32), reward, done, False, info


def test_real_episode_vec_monitor_skips_reset_pending_episodes(tmp_path):
    """The isolated-mode throwaway episode (done with reset_pending=True)
    must not become a monitor row or an info["episode"] record, and must
    not leak its placeholder steps into the next real episode's length."""
    script = [
        (1.0, True, {}),                        # real episode, r=1
        (0.0, False, {"reset_pending": True}),  # placeholder tick
        (0.0, True, {"reset_pending": True}),   # throwaway episode ends
        (2.0, True, {}),                        # real episode, r=2
    ]
    venv = DummyVecEnv([lambda: _ScriptedEnv(script)])
    mon = RealEpisodeVecMonitor(
        venv, filename=str(tmp_path / "monitor_test"))
    mon.reset()
    records = []
    for _ in range(len(script)):
        _, _, dones, infos = mon.step(np.array([0]))
        if dones[0]:
            records.append(infos[0].get("episode"))
    mon.close()

    assert records[0] is not None and records[0]["r"] == 1.0
    assert records[1] is None  # throwaway: no episode record for SB3
    assert records[2] is not None and records[2]["r"] == 2.0
    assert records[2]["l"] == 1  # placeholder steps didn't leak in

    csv = next(tmp_path.glob("monitor_test*.monitor.csv"))
    rows = [line for line in csv.read_text().splitlines()
            if line and not line.startswith("#") and not line.startswith("r,")]
    assert len(rows) == 2  # only the two real episodes hit the CSV
    assert [float(r.split(",")[0]) for r in rows] == [1.0, 2.0]


def test_make_env_wraps_with_async_resets_only_when_asked():
    with FakeGame([_episode()]) as fg:
        env = make_env(fg.port, async_resets=True)()
        assert isinstance(env, AsyncResetWrapper)
        env.close()
    with FakeGame([_episode()]) as fg:
        env = make_env(fg.port)()
        assert isinstance(env, HKEnv)  # default: no wrapper, N=1 unchanged
        env.close()
