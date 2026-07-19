import numpy as np
import pytest

from hkrl.env import ACTIONS, DEFAULT_REWARD, HKEnv
from hkrl.fake_game import FakeGame, obs, state


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
        _, r, terminated, truncated, info = env.step(0)
        assert terminated and not truncated and not info["won"]
        assert r < 0
        env.close()


def test_unknown_boss_state_maps_to_fallback_slot():
    episode = [state(obs(boss_state="Some Brand New Move")), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        o, _ = env.reset()
        assert not np.isnan(o).any()
        env.close()


def test_truncation_at_max_steps():
    # Episode that never ends (no done=True), but we truncate at max_steps.
    # This guards against the bug where truncated timeout is incorrectly
    # reported as terminated, which would corrupt PPO's value estimates.
    episode = [state(obs()), state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port, max_steps=1)
        env.reset()
        # After one step, we should hit max_steps truncation.
        _, _, terminated, truncated, info = env.step(0)
        assert truncated is True, "truncated should be True when max_steps is reached"
        assert terminated is False, "terminated should be False for a truncation (not a real terminal state)"
        env.close()


def test_terminal_step_reports_boss_damage_fraction():
    episode = [state(obs(bhp=900)), state(obs(bhp=675), done=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, _, done, _, info = env.step(0)
        env.close()
    assert done is True
    assert info["boss_damage_frac"] == pytest.approx(0.25)


def test_win_reports_full_boss_damage():
    episode = [state(obs(bhp=900)), state(obs(bhp=0), done=True, won=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, _, _, _, info = env.step(0)
        env.close()
    assert info["boss_damage_frac"] == pytest.approx(1.0)


def test_mid_episode_steps_do_not_carry_boss_damage():
    episode = [state(obs(bhp=900)), state(obs(bhp=800)), state(obs(bhp=700), done=True)]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        _, _, _, _, mid_info = env.step(0)
        _, _, _, _, end_info = env.step(0)
        env.close()
    assert "boss_damage_frac" not in mid_info
    assert end_info["boss_damage_frac"] == pytest.approx(200 / 900)


def test_truncation_reports_boss_damage_dealt_so_far():
    # No done=True frame: max_steps=1 forces truncation after one step, and
    # the damage dealt up to that point must still be reported.
    episode = [state(obs(bhp=900)), state(obs(bhp=450))]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port, max_steps=1)
        env.reset()
        _, _, done, truncated, info = env.step(0)
        env.close()
    assert done is False and truncated is True
    assert info["boss_damage_frac"] == pytest.approx(0.5)
