import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import train  # noqa: E402  (path insert must precede this import)

import pytest  # noqa: E402

from hkrl.fake_game import FakeGame, obs, state
from hkrl.generations import GenerationCallback, latest_checkpoint


def _won_episode(steps=6):
    """Every scripted episode ends in a win with the boss at 0 HP, so the
    manifest's win_rate and mean_boss_damage have known expected values."""
    frames = [state(obs(bhp=900))]
    frames += [state(obs(bhp=900)) for _ in range(steps - 1)]
    frames.append(state(obs(bhp=0), done=True, won=True))
    return frames


def _episodes(n):
    return [_won_episode() for _ in range(n)]


def test_a_short_training_run_writes_generations_and_a_manifest(tmp_path):
    with FakeGame(_episodes(40)) as fg:
        env, supervisor = train.build_env([fg.port], relaunch=lambda s: None,
                                          run_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=8,
                                    supervisor=supervisor)
            model.learn(total_timesteps=16, callback=cb)
        finally:
            env.close()

    gens = [json.loads(line)
            for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [1, 2]
    assert gens[-1]["timestep"] == 16
    for g in gens:
        assert g["recoveries"] == 0
        assert g["episodes"] >= 1  # 6-step episodes finish inside each 8-step window
        assert g["win_rate"] == 1.0  # enrichment read won=True from raw infos
        assert g["mean_boss_damage"] == 1.0
    gen, weights, vecnorm = latest_checkpoint(tmp_path)
    assert gen == 2 and weights.exists() and vecnorm.exists()
    assert list(tmp_path.glob("monitor_*")) != []  # VecMonitor session file


def test_two_instance_training_collects_from_both_games(tmp_path):
    """--instances N end to end at N=2 (minus the real processes): two
    bridges feed one vectorized PPO through build_env, and the rollout
    stripes across the whole fleet rather than draining one game."""
    with FakeGame(_episodes(40)) as a, FakeGame(_episodes(40)) as b:
        env, supervisor = train.build_env([a.port, b.port],
                                          relaunch=lambda s: None,
                                          run_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=16,
                                    supervisor=supervisor)
            # n_steps is per env, so one rollout is 16 timesteps; 32 makes
            # two full rollouts and two generations.
            model.learn(total_timesteps=32, callback=cb)
        finally:
            env.close()
        # Scripted episodes are consumed per instance (FakeGame pops them),
        # so both shrinking proves both games actually served the rollout.
        assert len(a.episodes) < 40
        assert len(b.episodes) < 40

    gens = [json.loads(line)
            for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [1, 2]
    assert gens[-1]["timestep"] == 32
    for g in gens:
        # The manifest aggregates across the fleet: every scripted episode
        # is a win at full boss damage on both instances.
        assert g["episodes"] >= 2
        assert g["win_rate"] == 1.0
        assert g["mean_boss_damage"] == 1.0


def test_default_n_steps_keeps_the_total_batch_constant():
    # 2048 total whatever the fleet size: the PPO update runs while every
    # game connection idles, and the mod severs connections idle for 10s,
    # so the update's wall-clock time must not grow with --instances.
    assert train.default_n_steps(1) == 2048
    assert train.default_n_steps(2) == 1024
    assert train.default_n_steps(4) == 512
    assert train.default_n_steps(1000) == 128  # floored, never zero


def test_stop_flag_ends_training_at_the_current_episodes_end(tmp_path):
    """A stop request finishes the attempt in progress rather than cutting
    the fight off mid-swing: stopping at the episode boundary also leaves
    the game in a state the next session's reset handles cheaply, instead
    of a mid-fight truncation the reset macro has to unwind."""
    flag = threading.Event()
    flag.set()
    with FakeGame(_episodes(5)) as fg:
        env, _ = train.build_env([fg.port], relaunch=lambda s: None,
                                 run_dir=tmp_path)
        try:
            model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
            model.learn(total_timesteps=16, callback=train.StopOnFlag(flag))
        finally:
            env.close()
    # The flag was set before the first step, so training ran exactly one
    # scripted 6-step episode -- not zero steps, and not the full rollout.
    assert model.num_timesteps == 6


def test_resume_continues_timesteps_norm_stats_and_generation_numbering(tmp_path):
    with FakeGame(_episodes(40)) as fg:
        env, supervisor = train.build_env([fg.port], relaunch=lambda s: None,
                                          run_dir=tmp_path)
        model = train.build_model(env, tmp_path, n_steps=8, batch_size=8)
        cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=8,
                                supervisor=supervisor)
        model.learn(total_timesteps=16, callback=cb)
        saved_count = env.obs_rms.count
        env.close()

    gen, weights, vecnorm = latest_checkpoint(tmp_path)
    assert gen == 2

    with FakeGame(_episodes(40)) as fg:
        env, supervisor = train.build_env([fg.port], relaunch=lambda s: None,
                                          run_dir=tmp_path, resume_vecnorm=vecnorm)
        try:
            # The statistics were loaded, not freshly initialized: a fresh
            # VecNormalize starts its count at epsilon (1e-4).
            assert env.obs_rms.count == pytest.approx(saved_count)
            model = train.build_model(env, tmp_path, resume_model=weights)
            assert model.num_timesteps == 16  # resumed, not restarted
            cb = GenerationCallback(tmp_path, vecnorm=env, every_steps=8,
                                    supervisor=supervisor)
            model.learn(total_timesteps=8, callback=cb,
                        reset_num_timesteps=False)
            assert model.num_timesteps == 24
        finally:
            env.close()

    gens = [json.loads(line)
            for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [g["gen"] for g in gens] == [1, 2, 3]
    assert gens[-1]["timestep"] == 24
