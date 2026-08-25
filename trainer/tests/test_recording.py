import gzip
import json

from hkrl.bosses import BOSSES
from hkrl.env import ACTIONS, DEFAULT_REWARD, OBS_KEYS
from hkrl.protocol import PROTOCOL_VERSION
from hkrl.recording import (RecordingWriter, SCHEMA_VERSION, build_header,
                            read_recording, recording_path)


def test_recording_path_shape(tmp_path):
    p = recording_path(tmp_path / "replays", gen=7)
    assert p.parent == tmp_path / "replays"
    assert p.name.endswith("_gen0007.jsonl.gz")
    # UTC timestamp prefix: yyyymmdd-HHMMSS
    stamp = p.name.split("_gen")[0]
    assert len(stamp) == 15 and stamp[8] == "-" and stamp[:8].isdigit()


def test_writer_round_trip(tmp_path):
    path = tmp_path / "rec.jsonl.gz"
    with RecordingWriter(path) as w:
        w.header(run_id="hornet-1", gen=3)
        w.step(ep=1, i=0, a=5, r=-0.001)
        w.step(ep=1, i=1, a=6, r=1.0)
        w.episode(ep=1, result="WIN", steps=2, reward=0.999)
    rows = read_recording(path)
    assert [r["type"] for r in rows] == ["header", "step", "step", "episode"]
    assert rows[0]["schema_version"] == SCHEMA_VERSION
    assert rows[0]["run_id"] == "hornet-1"
    assert rows[1]["a"] == 5 and rows[2]["i"] == 1
    assert rows[3]["result"] == "WIN"


def test_interrupted_recording_keeps_a_parseable_prefix(tmp_path):
    # Simulate a Ctrl-C: rows written and flushed, but the gzip stream never
    # closed (no trailer). The bytes on disk at that moment must still parse.
    path = tmp_path / "rec.jsonl.gz"
    w = RecordingWriter(path)
    w.header(run_id="hornet-1", gen=3)
    w.step(ep=1, i=0, a=5, r=-0.001)
    snapshot = tmp_path / "killed.jsonl.gz"
    snapshot.write_bytes(path.read_bytes())  # what an interrupt leaves behind
    w.close()
    rows = read_recording(snapshot)
    assert [r["type"] for r in rows] == ["header", "step"]
    assert rows[1]["a"] == 5


def test_build_header_freezes_the_label_maps(tmp_path):
    run_dir = tmp_path / "gruz-1"
    h = build_header(run_dir=run_dir, gen=12, weights=run_dir / "gen_0012.zip",
                     vecnorm=run_dir / "gen_0012_vecnorm.pkl",
                     boss_id="gruz_mother", deterministic=True, episodes=5)
    assert h["run_id"] == "gruz-1" and h["gen"] == 12
    assert h["weights"] == "gen_0012.zip" and h["vecnorm"] == "gen_0012_vecnorm.pkl"
    assert h["boss"] == "gruz_mother"
    assert h["boss_spec"]["display_name"] == "Gruz Mother"
    assert list(h["boss_spec"]["fsm_states"]) == list(BOSSES["gruz_mother"].fsm_states)
    assert h["boss_spec"]["arena_half_w"] == BOSSES["gruz_mother"].arena_half_w
    assert h["actions"] == ACTIONS and h["obs_keys"] == OBS_KEYS
    assert h["reward_config"] == DEFAULT_REWARD
    assert h["protocol_version"] == PROTOCOL_VERSION
    assert h["deterministic"] is True and h["episodes_requested"] == 5
    assert h["timescale"] == 1.0 and h["headless"] is False and h["auto"] is False
    assert set(h["versions"]) == {"sb3_contrib", "stable_baselines3", "torch",
                                  "python"}
    json.dumps(h)  # everything must be JSON-serializable as-is
