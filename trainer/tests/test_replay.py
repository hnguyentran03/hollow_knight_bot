import pathlib
import sys

import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import replay  # noqa: E402  (path insert must precede this import)

from hkrl.fake_game import FakeGame, obs, state
from hkrl.vec import make_env


def _scripted(win):
    """A 4-step episode with a known outcome: reset frame, three mid frames,
    then a terminal frame -- boss at 0 HP on a win, knight dead on a loss
    with half the boss's HP removed."""
    frames = [state(obs(bhp=900))] + [state(obs(bhp=900)) for _ in range(3)]
    if win:
        frames.append(state(obs(bhp=0), done=True, won=True))
    else:
        frames.append(state(obs(bhp=450, khp=0), done=True))
    return frames


def _make_checkpoint(tmp_path):
    """A real, loadable checkpoint: an untrained PPO with this pipeline's
    spaces, plus the VecNormalize statistics it was constructed with."""
    with FakeGame([_scripted(win=False)]) as fg:
        venv = DummyVecEnv([make_env(fg.port)])
        stacked = VecFrameStack(venv, n_stack=4)
        env = VecNormalize(stacked, gamma=0.995)
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu")
        weights = tmp_path / "gen_0001.zip"
        vecnorm = tmp_path / "gen_0001_vecnorm.pkl"
        model.save(weights)
        env.save(str(vecnorm))
        env.close()
    return weights, vecnorm


def test_replay_reports_per_episode_stats(tmp_path, capsys):
    weights, vecnorm = _make_checkpoint(tmp_path)
    # One extra scripted episode: DummyVecEnv autoresets on the final terminal step, eagerly consuming one more reset() before replay()'s loop exits.
    with FakeGame([_scripted(win=True), _scripted(win=False), _scripted(win=False)]) as fg:
        model, env = replay.load_policy(weights, vecnorm, port=fg.port)
        try:
            summaries = replay.replay(model, env, episodes=2)
        finally:
            env.close()

    assert [s["result"] for s in summaries] == ["WIN", "loss"]
    assert summaries[0]["won"] is True
    assert summaries[0]["steps"] == 4
    assert summaries[0]["boss_damage_frac"] == pytest.approx(1.0)
    assert summaries[1]["boss_damage_frac"] == pytest.approx(0.5)
    out = capsys.readouterr().out
    assert "result=WIN" in out and "boss_dmg=" in out
    # Rewards are printed unnormalized: a win's +10 terminal bonus dominates.
    assert summaries[0]["reward"] > 5.0


def test_deterministic_replay_reproduces_itself(tmp_path):
    # The frozen-statistics + deterministic-argmax pipeline must be
    # repeatable: identical scripted games produce identical summaries.
    weights, vecnorm = _make_checkpoint(tmp_path)
    results = []
    for _ in range(2):
        # One extra scripted episode: DummyVecEnv autoresets on the final terminal step, eagerly consuming one more reset() before replay()'s loop exits.
        with FakeGame([_scripted(win=False), _scripted(win=False)]) as fg:
            model, env = replay.load_policy(weights, vecnorm, port=fg.port)
            try:
                results.append(replay.replay(model, env, episodes=1))
            finally:
                env.close()
    assert results[0] == results[1]
