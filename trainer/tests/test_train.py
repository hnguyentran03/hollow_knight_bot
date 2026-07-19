import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import train  # noqa: E402  (path insert must precede this import)

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


def test_stop_flag_ends_training_at_the_next_step(tmp_path):
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
    assert model.num_timesteps <= 2
