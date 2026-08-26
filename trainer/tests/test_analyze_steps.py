import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import analyze_steps  # noqa: E402  (path insert must precede this import)

from hkrl.env import ACTIONS  # noqa: E402
from hkrl.recording import RecordingWriter, read_recording  # noqa: E402


_ARENA = {"arena_center_x": 26.5, "arena_half_w": 11.23,
          "floor_y": 28.41, "arena_height": 9.59}


def _step_defaults(i, state="Antic", action=0):
    obs = {"kx": 26.5, "ky": 28.5, "kvx": 0.0, "kvy": 0.0, "khp": 9,
           "soul": 0, "on_ground": True, "dashing": False, "invuln": False,
           "facing_right": True, "bx": 30.0, "by": 28.5, "bvx": 0.0,
           "bvy": 0.0, "bhp": 900, "boss_state": state,
           "projectile_active": False, "px": 0.0, "py": 0.0}
    pi = [0.0] * len(ACTIONS)
    pi[action] = 0.8
    pi[(action + 1) % len(ACTIONS)] = 0.2
    return dict(ep=1, i=i, obs=obs, a=action, pi=pi, v=0.5, logp=-0.2,
                ent=1.0, h_norm=1.0, r=0.0, r_terms={}, done=False,
                trunc=False, won=False)


def _write_recording(path, boss="gorb", steps=(), episodes=None, gen=7):
    """A schema-v1 file from compact scripts. Each entry of `steps` is
    either a (boss_state, action) tuple or a dict of overrides merged over
    _step_defaults (an "obs" override updates, not replaces, the default
    obs; an "a" override without an explicit "pi" re-peaks pi on it).
    `episodes` lists episode-summary override dicts; the default is a
    single loss episode covering all steps."""
    with RecordingWriter(path) as w:
        w.header(run_id="test-run", gen=gen, boss=boss,
                 boss_spec={"id": boss, "display_name": boss.title(),
                            "fsm_states": ["Wait", "Antic", "Attack",
                                           "UNKNOWN"], **_ARENA},
                 actions=ACTIONS, obs_keys=["kx"], deterministic=True,
                 episodes_requested=1)
        n = 0
        for i, entry in enumerate(steps):
            if isinstance(entry, tuple):
                row = _step_defaults(i, state=entry[0], action=entry[1])
            else:
                entry = dict(entry)
                row = _step_defaults(i)
                obs_over = entry.pop("obs", {})
                row.update(entry)
                row["obs"] = {**row["obs"], **obs_over}
                if "a" in entry and "pi" not in entry:
                    pi = [0.0] * len(ACTIONS)
                    pi[row["a"]] = 0.8
                    pi[(row["a"] + 1) % len(ACTIONS)] = 0.2
                    row["pi"] = pi
            w.step(**row)
            n += 1
        for epi in (episodes if episodes is not None else [{}] if n else []):
            summary = dict(ep=1, result="loss", steps=n, reward=0.0,
                           boss_damage_frac=0.0, attempt=1, wall_s=1.0)
            summary.update(epi)
            w.episode(**summary)


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


def test_action_class_priority_and_coverage():
    from hkrl.analysis import ACTION_CLASSES, action_class
    # Every frozen action maps into the fixed class list.
    classes = [action_class(a) for a in ACTIONS]
    assert set(classes) <= set(ACTION_CLASSES)
    assert classes[0] == "idle"                  # no buttons
    assert classes[1] == classes[2] == "move"    # bare directions
    assert classes[3] == "jump"
    assert classes[6] == "attack"
    assert classes[12] == "attack"               # ↑Atk: attack, not move
    assert classes[14] == "attack"               # Jump+Atk: attack wins
    assert classes[9] == "dash"
    assert classes[15] == "cast"
    assert classes[20] == "focus"


def test_episode_results_maps_ep_to_summary(tmp_path):
    from hkrl.analysis import episode_results
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[("Antic", 5)])
    results = episode_results(read_recording(rec))
    assert results[1]["result"] == "loss"
    assert episode_results([]) == {}


def test_confidence_trace_series_and_events(tmp_path):
    from hkrl.analysis import confidence_trace
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[
        {"i": 0, "a": 5, "ent": 3.0446, "v": 1.0},          # ~ln(21): ent→1.0
        {"i": 1, "a": 5, "r_terms": {"knight_hit": -1.0},
         "obs": {"khp": 8}},
        {"i": 2, "a": 6, "r_terms": {"boss_damage": 0.6}},
        {"i": 3, "a": 6, "r_terms": {"win": 10.0}, "done": True,
         "won": True},
    ], episodes=[{"result": "WIN"}])
    (ep,) = confidence_trace(read_recording(rec))
    assert ep["ep"] == 1 and ep["result"] == "WIN" and ep["steps"] == 4
    assert len(ep["pia"]) == len(ep["ent"]) == len(ep["v"]) == 4
    assert ep["pia"][0] == pytest.approx(0.8)
    assert 0.99 <= ep["ent"][0] <= 1.0            # normalized by ln(21)
    assert ep["khp"][1] == 8
    kinds = {(e["i"], e["kind"]) for e in ep["events"]}
    assert kinds == {(1, "hit"), (2, "dealt"), (3, "win")}


def test_confidence_trace_marks_interrupted_and_terminal_kinds(tmp_path):
    from hkrl.analysis import confidence_trace
    rec = tmp_path / "r.jsonl.gz"
    # ep 1 dies; ep 2 has steps but no episode summary (interrupt).
    _write_recording(rec, steps=[
        {"i": 0, "a": 1, "r_terms": {"death": -5.0}, "done": True},
        {"ep": 2, "i": 0, "a": 1},
    ], episodes=[{"result": "loss", "steps": 1}])
    eps = confidence_trace(read_recording(rec))
    assert [e["ep"] for e in eps] == [1, 2]
    assert eps[0]["events"] == [{"i": 0, "kind": "death"}]
    assert eps[1]["result"] is None and eps[1]["events"] == []


def test_render_writes_a_png(tmp_path):
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[("Antic", 5), ("Wait", 1), ("Antic", 6)])
    rows = read_recording(rec)
    header, steps, episodes = analyze_steps.merge_recordings([rows])
    agg = analyze_steps.aggregate(rows, min_steps=1)
    out = tmp_path / "matrix.png"
    analyze_steps.render(agg, header, episodes, out)
    assert out.exists() and out.stat().st_size > 1000
