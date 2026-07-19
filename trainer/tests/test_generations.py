import json
from pathlib import Path

import pytest

from hkrl.generations import (
    GenerationCallback, checkpoint_paths, last_generation, latest_checkpoint,
    record_generation, summarize_episodes,
)


def _record(tmp_path, gen, timestep=100):
    record_generation(
        tmp_path, gen=gen, timestep=timestep, wall_clock_s=12.34,
        stats={"episodes": 2, "mean_reward": -4.0, "win_rate": 0.5,
               "mean_episode_len": 280.0, "mean_boss_damage": 0.05},
        recoveries=1,
        save_model=lambda p: Path(p).write_text("weights"),
        save_vecnorm=lambda p: Path(p).write_text("stats"),
    )


def test_record_generation_writes_weights_norm_stats_and_a_manifest_line(tmp_path):
    _record(tmp_path, gen=1, timestep=10)
    weights, vecnorm = checkpoint_paths(tmp_path, 1)
    assert weights.name == "gen_0001.zip"
    assert weights.read_text() == "weights"
    # The normalization statistics must travel with the weights: a policy
    # loaded without them sees a different observation distribution.
    assert vecnorm.read_text() == "stats"
    line = json.loads((tmp_path / "generations.jsonl").read_text().strip())
    assert line["gen"] == 1
    assert line["timestep"] == 10
    assert line["mean_boss_damage"] == 0.05
    assert line["recoveries"] == 1
    assert line["wall_clock_s"] == pytest.approx(12.3, abs=0.1)


def test_manifest_appends_and_last_generation_tracks_it(tmp_path):
    assert last_generation(tmp_path) == 0
    _record(tmp_path, gen=1)
    _record(tmp_path, gen=2)
    assert len((tmp_path / "generations.jsonl").read_text().splitlines()) == 2
    assert last_generation(tmp_path) == 2


def test_latest_checkpoint_returns_the_newest_complete_pair(tmp_path):
    _record(tmp_path, gen=1)
    _record(tmp_path, gen=2)
    gen, weights, vecnorm = latest_checkpoint(tmp_path)
    assert gen == 2
    assert (weights, vecnorm) == checkpoint_paths(tmp_path, 2)

    # A generation whose files were deleted is skipped, not returned broken.
    for p in checkpoint_paths(tmp_path, 2):
        p.unlink()
    gen, _, _ = latest_checkpoint(tmp_path)
    assert gen == 1


def test_latest_checkpoint_raises_when_the_run_has_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(tmp_path)


def test_summarize_episodes_aggregates_monitor_records():
    episodes = [
        {"r": -2.0, "l": 100, "won": True, "boss_damage_frac": 1.0},
        {"r": -6.0, "l": 300, "won": False, "boss_damage_frac": 0.2},
    ]
    assert summarize_episodes(episodes) == {
        "episodes": 2, "mean_reward": -4.0, "win_rate": 0.5,
        "mean_episode_len": 200.0, "mean_boss_damage": 0.6,
    }


def test_summarize_episodes_handles_an_empty_generation():
    assert summarize_episodes([]) == {
        "episodes": 0, "mean_reward": 0.0, "win_rate": 0.0,
        "mean_episode_len": 0.0, "mean_boss_damage": 0.0,
    }


def test_callback_numbering_continues_from_the_manifest(tmp_path):
    # Resume must extend the sequence, not restart it at gen 1 and
    # overwrite earlier checkpoints.
    _record(tmp_path, gen=1)
    _record(tmp_path, gen=2)
    cb = GenerationCallback(tmp_path, vecnorm=None)
    assert cb._gen == 2
