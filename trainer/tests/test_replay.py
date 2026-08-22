import json
import pathlib
import sys

import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    """A real, loadable checkpoint: an untrained RecurrentPPO with this
    pipeline's spaces, plus the VecNormalize statistics it was constructed
    with. No VecFrameStack -- the recurrent policy replaced frame stacking,
    so replay's pipeline (which this must match) no longer stacks either."""
    with FakeGame([_scripted(win=False)]) as fg:
        venv = DummyVecEnv([make_env(fg.port)])
        env = VecNormalize(venv, gamma=0.995)
        model = RecurrentPPO("MlpLstmPolicy", env, n_steps=8, batch_size=8,
                             device="cpu")
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
        model, env = replay.load_policy(weights, vecnorm, port=fg.port,
                                        run_dir=tmp_path)
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


def test_banner_names_generation_run_and_episodes():
    text = replay.banner(3, pathlib.Path("/x/runs/r1"), 5)
    assert "generation 3" in text
    assert "/x/runs/r1" in text
    assert "5 episodes" in text


def test_auto_and_root_are_accepted_flags():
    # --auto makes replay self-contained (launches its own game); --root is
    # where it backs up the master save. Both parse WITHOUT touching a game.
    args = replay.build_parser().parse_args(
        ["--run-dir", "/x", "--auto", "--gen", "2", "--episodes", "4",
         "--root", "/tmp/hk"])
    assert args.auto is True
    assert args.gen == 2 and args.episodes == 4
    assert str(args.root) == "/tmp/hk"


def test_auto_defaults_off_and_root_has_a_default():
    args = replay.build_parser().parse_args(["--run-dir", "/x"])
    assert args.auto is False        # unchanged: connect to a running game
    assert args.root is not None     # ~/hkrl by default, for backup_saves


def test_headless_and_timescale_are_accepted_flags():
    # --auto --headless --timescale K is the speed-fidelity gate command;
    # both must parse without touching a game.
    args = replay.build_parser().parse_args(
        ["--run-dir", "x", "--auto", "--headless", "--timescale", "2"])
    assert args.headless is True
    assert args.timescale == 2.0
    args = replay.build_parser().parse_args(["--run-dir", "x"])
    assert args.headless is False
    assert args.timescale == 1.0


def test_replay_stops_at_the_episode_boundary_when_flagged(tmp_path):
    # A set stop flag ends the loop at the next episode boundary (mirrors
    # train.py's StopOnFlag): the dashboard's single Stop -> SIGINT sets it,
    # and the in-progress episode still finishes rather than being severed.
    import threading
    weights, vecnorm = _make_checkpoint(tmp_path)
    stop = threading.Event()
    stop.set()
    with FakeGame([_scripted(win=True), _scripted(win=False)]) as fg:
        model, env = replay.load_policy(weights, vecnorm, port=fg.port,
                                        run_dir=tmp_path)
        try:
            summaries = replay.replay(model, env, episodes=5, stop=stop)
        finally:
            env.close()
    assert summaries == []  # already stopped: not even the first episode ran


def test_deterministic_replay_reproduces_itself(tmp_path):
    # The frozen-statistics + deterministic-argmax pipeline must be
    # repeatable: identical scripted games produce identical summaries.
    weights, vecnorm = _make_checkpoint(tmp_path)
    results = []
    for _ in range(2):
        # One extra scripted episode: DummyVecEnv autoresets on the final terminal step, eagerly consuming one more reset() before replay()'s loop exits.
        with FakeGame([_scripted(win=False), _scripted(win=False)]) as fg:
            model, env = replay.load_policy(weights, vecnorm, port=fg.port,
                                            run_dir=tmp_path)
            try:
                results.append(replay.replay(model, env, episodes=1))
            finally:
                env.close()
    assert results[0] == results[1]


def test_run_boss_reads_config_and_defaults_to_hornet1(tmp_path):
    assert replay.run_boss(tmp_path) == "hornet1"          # no config at all
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"boss": "gruz_mother"}) + "\n")
    assert replay.run_boss(tmp_path) == "gruz_mother"
