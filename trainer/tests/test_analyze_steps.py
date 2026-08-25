import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import analyze_steps  # noqa: E402  (path insert must precede this import)

from hkrl.env import ACTIONS  # noqa: E402
from hkrl.recording import RecordingWriter, read_recording  # noqa: E402


def _write_recording(path, boss="gorb", steps=()):
    """A minimal schema-v1 file: header + the given (state, action, pi) steps.

    pi defaults to a distribution peaked on the chosen action so the mean-pi
    matrix is predictable without hand-writing 21 floats per step."""
    with RecordingWriter(path) as w:
        w.header(run_id="test-run", gen=7, boss=boss,
                 boss_spec={"id": boss, "display_name": boss.title(),
                            "fsm_states": ["Wait", "Antic", "Attack",
                                           "UNKNOWN"]},
                 actions=ACTIONS, obs_keys=["kx"], deterministic=True,
                 episodes_requested=1)
        for i, (state, action) in enumerate(steps):
            pi = [0.0] * len(ACTIONS)
            pi[action] = 0.8
            pi[(action + 1) % len(ACTIONS)] = 0.2
            w.step(ep=1, i=i, obs={"boss_state": state}, a=action, pi=pi,
                   r=0.0, r_terms={}, done=False, trunc=False, won=False)
        w.episode(ep=1, result="loss", steps=len(steps), reward=0.0,
                  boss_damage_frac=0.0, attempt=1, wall_s=1.0)


def test_aggregate_orders_states_by_frequency_and_averages_pi(tmp_path):
    rec = tmp_path / "r.jsonl.gz"
    # Antic appears 3x (always action 5), Wait 2x (actions 1 then 2).
    _write_recording(rec, steps=[("Antic", 5), ("Wait", 1), ("Antic", 5),
                                 ("Wait", 2), ("Antic", 5)])
    agg = analyze_steps.aggregate(read_recording(rec), min_steps=1)
    assert agg.states == ["Antic", "Wait"]          # frequency order
    assert agg.counts == [3, 2]
    assert agg.matrix[0][5] == pytest.approx(0.8)   # mean pi over Antic rows
    for row in agg.matrix:                          # rows stay distributions
        assert sum(row) == pytest.approx(1.0)
    assert agg.modal == [5, 1]                      # tie in Wait -> first id


def test_aggregate_drops_states_below_min_steps(tmp_path):
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[("Antic", 5)] * 5 + [("Attack", 0)])
    agg = analyze_steps.aggregate(read_recording(rec), min_steps=2)
    assert agg.states == ["Antic"]
    assert agg.dropped == [("Attack", 1)]


def test_merge_recordings_rejects_mixed_bosses(tmp_path):
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    _write_recording(a, boss="gorb", steps=[("Antic", 5)])
    _write_recording(b, boss="marmu", steps=[("Chase", 1)])
    with pytest.raises(ValueError, match="gorb.*marmu|marmu.*gorb"):
        analyze_steps.merge_recordings([read_recording(a), read_recording(b)])


def test_merge_recordings_concatenates_steps_and_sums_episodes(tmp_path):
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    _write_recording(a, steps=[("Antic", 5)] * 2)
    _write_recording(b, steps=[("Antic", 5), ("Wait", 1)])
    header, steps, episodes = analyze_steps.merge_recordings(
        [read_recording(a), read_recording(b)])
    assert header["boss"] == "gorb"
    assert len(steps) == 4
    assert episodes == 2


def test_action_labels_compact_and_complete():
    labels = analyze_steps.action_labels(ACTIONS)
    assert len(labels) == len(ACTIONS) == 21
    assert labels[0] == "·"                     # no buttons held
    assert labels[1] == "←" and labels[2] == "→"
    assert labels[4] == "←Jump"
    assert labels[12] == "↑Atk" and labels[13] == "↓Atk"
    assert labels[14] == "JumpAtk"
    assert labels[20] == "Focus"
    assert len(set(labels)) == len(labels)      # no ambiguous duplicates


def test_render_writes_a_png(tmp_path):
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[("Antic", 5), ("Wait", 1), ("Antic", 6)])
    rows = read_recording(rec)
    header, steps, episodes = analyze_steps.merge_recordings([rows])
    agg = analyze_steps.aggregate(rows, min_steps=1)
    out = tmp_path / "matrix.png"
    analyze_steps.render(agg, header, episodes, out)
    assert out.exists() and out.stat().st_size > 1000
