import numpy as np

from hkrl.env import ACTIONS, DEFAULT_REWARD, HKEnv
from hkrl.fake_game import FakeGame


def obs(kx=20.0, khp=9, bhp=900, boss_state="Idle", **kw):
    base = {"kx": kx, "ky": 6.0, "kvx": 0.0, "kvy": 0.0, "khp": khp, "soul": 0,
            "on_ground": True, "dashing": False, "invuln": False, "facing_right": True,
            "bx": 30.0, "by": 6.0, "bvx": 0.0, "bvy": 0.0, "bhp": bhp,
            "boss_state": boss_state, "needle_active": False, "nx": 0.0, "ny": 0.0}
    base.update(kw)
    return base


def state(o, done=False, won=False):
    return {"type": "state", "obs": o, "done": done,
            "info": {"won": won, "scene": "GG_Hornet_1", "attempt": 1}}


def test_action_space_and_obs_shape():
    episode = [state(obs()), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        o, info = env.reset()
        assert env.action_space.n == len(ACTIONS) == 15
        assert o.shape == env.observation_space.shape
        assert o.dtype == np.float32
        env.close()


def test_reward_for_boss_damage_and_knight_damage():
    episode = [state(obs(bhp=900)), state(obs(bhp=880)), state(obs(bhp=880, khp=8))]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, r1, *_ = env.step(0)
        assert r1 == (20 * DEFAULT_REWARD["boss_hp_scale"]
                      + DEFAULT_REWARD["time_penalty"])
        _, r2, *_ = env.step(0)
        assert r2 == (DEFAULT_REWARD["knight_hit"] + DEFAULT_REWARD["time_penalty"])
        env.close()


def test_win_and_loss_bonuses_and_termination():
    win_ep = [state(obs()), state(obs(bhp=0), done=True, won=True)]
    loss_ep = [state(obs()), state(obs(khp=0), done=True, won=False)]
    with FakeGame([win_ep, loss_ep]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, r, terminated, truncated, info = env.step(0)
        assert terminated and not truncated and info["won"]
        assert r > DEFAULT_REWARD["win"] / 2  # dominated by the win bonus
        env.reset()
        _, r, terminated, _, info = env.step(0)
        assert terminated and not info["won"]
        assert r < 0
        env.close()


def test_unknown_boss_state_maps_to_fallback_slot():
    episode = [state(obs(boss_state="Some Brand New Move")), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        o, _ = env.reset()
        assert not np.isnan(o).any()
        env.close()
