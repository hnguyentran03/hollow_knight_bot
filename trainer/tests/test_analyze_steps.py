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


def test_arena_occupancy_bins_splits_and_marks_deaths(tmp_path):
    from hkrl.analysis import arena_occupancy
    rec = tmp_path / "r.jsonl.gz"
    x_left, y_floor = 26.5 - 11.23, 28.41
    _write_recording(rec, steps=[
        # ep 1 wins standing at the left-bottom corner.
        {"i": 0, "obs": {"kx": x_left + 0.1, "ky": y_floor + 0.1}},
        {"i": 1, "obs": {"kx": x_left + 0.1, "ky": y_floor + 0.1},
         "done": True, "won": True},
        # ep 2 dies out of bounds right (clamps to edge bin).
        {"ep": 2, "i": 0, "obs": {"kx": 99.0, "ky": y_floor + 0.1},
         "done": True},
        # ep 3 times out mid-arena: occupancy with the losses, no death.
        # (27.0 sits mid-bin; the exact arena center is a bin edge and
        # float error makes its index platform-shaky)
        {"ep": 3, "i": 0, "obs": {"kx": 27.0, "ky": y_floor + 0.1},
         "trunc": True},
    ], episodes=[{"result": "WIN", "steps": 2},
                 {"ep": 2, "result": "loss", "steps": 1},
                 {"ep": 3, "result": "TIMEOUT", "steps": 1}])
    a = arena_occupancy(read_recording(rec))
    assert (a["nx"], a["ny"]) == (24, 12)
    assert a["win"]["episodes"] == 1 and a["loss"]["episodes"] == 2
    assert a["win"]["grid"][11][0] == pytest.approx(1.0)   # bottom row = iy 11
    assert sum(map(sum, a["win"]["grid"])) == pytest.approx(1.0)
    assert sum(map(sum, a["loss"]["grid"])) == pytest.approx(1.0)
    assert a["loss"]["deaths"] == [[23, 11]]               # clamped edge bin
    assert a["loss"]["grid"][11][12] > 0                   # timeout occupancy


def test_arena_occupancy_handles_a_single_outcome_side(tmp_path):
    from hkrl.analysis import arena_occupancy
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[{"i": 0, "done": True, "won": True}],
                     episodes=[{"result": "WIN", "steps": 1}])
    a = arena_occupancy(read_recording(rec))
    assert a["loss"]["episodes"] == 0 and a["loss"]["deaths"] == []
    assert sum(map(sum, a["loss"]["grid"])) == 0.0


def test_postmortems_selects_loss_windows(tmp_path):
    from hkrl.analysis import postmortems
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=(
        [{"i": i} for i in range(4)]
        + [{"i": 4, "r_terms": {"knight_hit": -1.0, "death": -5.0},
            "done": True}]
        + [{"ep": 2, "i": 0, "done": True, "won": True}]
        + [{"ep": 3, "i": 0, "trunc": True}]),
        episodes=[{"result": "loss", "steps": 5},
                  {"ep": 2, "result": "WIN", "steps": 1},
                  {"ep": 3, "result": "TIMEOUT", "steps": 1}])
    pms = postmortems(read_recording(rec), window=3)
    assert [p["ep"] for p in pms] == [1]         # wins/timeouts excluded
    p0 = pms[0]
    assert [s["i"] for s in p0["steps"]] == [2, 3, 4]   # window truncates
    assert p0["total_steps"] == 5
    assert p0["killing_terms"] == {"knight_hit": -1.0, "death": -5.0}


def test_reaction_profile_detects_onsets_and_buckets(tmp_path):
    from hkrl.analysis import reaction_profile
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[
        # Onset at i=1, |px-kx| = 2 (near); next 2 steps dash (a=9).
        {"i": 0},
        {"i": 1, "a": 9, "obs": {"projectile_active": True, "px": 28.5}},
        {"i": 2, "a": 9, "obs": {"projectile_active": True, "px": 27.5}},
        # Still active: no second onset.
        {"i": 3, "obs": {"projectile_active": True, "px": 27.0}},
        # ep 2 starts already active at |px-kx| = 10 (far): counts.
        {"ep": 2, "i": 0, "a": 3,
         "obs": {"projectile_active": True, "px": 36.5}},
    ], episodes=[{"result": "loss", "steps": 4},
                 {"ep": 2, "result": "loss", "steps": 1}])
    prof = reaction_profile(read_recording(rec), window=2)
    assert prof["onsets"] == 2
    near = next(b for b in prof["buckets"] if b["name"] == "near")
    far = next(b for b in prof["buckets"] if b["name"] == "far")
    assert near["n"] == 1 and near["shares"]["dash"] == pytest.approx(1.0)
    assert far["n"] == 1 and far["shares"]["jump"] == pytest.approx(1.0)


def test_reaction_profile_no_onsets(tmp_path):
    from hkrl.analysis import reaction_profile
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[("Antic", 5)] * 3)
    assert reaction_profile(read_recording(rec))["onsets"] == 0


def test_soul_economy_buckets_and_shares(tmp_path):
    from hkrl.analysis import soul_economy
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[
        {"i": 0, "a": 15, "obs": {"khp": 5, "soul": 33}},   # cast @ 33
        {"i": 1, "a": 20, "obs": {"khp": 5, "soul": 33}},   # focus @ 33
        {"i": 2, "a": 0, "obs": {"khp": 5, "soul": 32}},    # idle @ 32
        {"i": 3, "a": 15, "obs": {"khp": 9, "soul": 99}},   # cast @ full
    ])
    econ = soul_economy(read_recording(rec))
    assert econ["soul_buckets"] == ["0–32", "33–65", "66–98", "99"]
    cell = econ["cells"][4][1]                   # khp 5 row, 33–65 bucket
    assert cell["n"] == 2
    assert cell["cast"] == pytest.approx(0.5)
    assert cell["focus"] == pytest.approx(0.5)
    assert econ["cells"][4][0] == {"n": 1, "cast": 0.0, "focus": 0.0}
    assert econ["cells"][8][3]["cast"] == pytest.approx(1.0)


def test_render_writes_a_png(tmp_path):
    rec = tmp_path / "r.jsonl.gz"
    _write_recording(rec, steps=[("Antic", 5), ("Wait", 1), ("Antic", 6)])
    rows = read_recording(rec)
    header, steps, episodes = analyze_steps.merge_recordings([rows])
    agg = analyze_steps.aggregate(rows, min_steps=1)
    out = tmp_path / "matrix.png"
    analyze_steps.render(agg, header, episodes, out)
    assert out.exists() and out.stat().st_size > 1000
