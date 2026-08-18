import json

import pytest

from hkrl.exports import export_generation, exported_generations


def _run(tmp_path, boss="gruz_mother", gens=(1, 2)):
    """A run dir with fake (byte-string) checkpoint pairs and a manifest
    line per gen -- export copies files, it never loads them."""
    run = tmp_path / "runs" / "r1"
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True)
    lines = []
    for g in gens:
        (ckpt / f"gen_{g:04d}.zip").write_bytes(b"w%d" % g)
        (ckpt / f"gen_{g:04d}_vecnorm.pkl").write_bytes(b"v%d" % g)
        lines.append(json.dumps({
            "gen": g, "timestep": g * 15000, "win_rate": 0.5,
            "mean_reward": 1.0, "episodes": 10,
            "mean_boss_damage": 0.4, "mean_episode_len": 400.0}))
    (run / "generations.jsonl").write_text("\n".join(lines) + "\n")
    if boss:
        (run / "config.jsonl").write_text(json.dumps({"boss": boss}) + "\n")
    return run


def test_exports_latest_gen_by_default_with_manifest(tmp_path):
    run = _run(tmp_path)
    dest = export_generation(tmp_path, run)
    assert dest == tmp_path / "exports" / "r1_gen0002"
    assert (dest / "model.zip").read_bytes() == b"w2"
    assert (dest / "vecnorm.pkl").read_bytes() == b"v2"
    m = json.loads((dest / "manifest.json").read_text())
    assert m["name"] == "r1_gen0002"
    assert m["run_id"] == "r1" and m["gen"] == 2
    assert m["timestep"] == 30000
    assert m["boss"] == "gruz_mother"
    assert m["boss_display"] == "Gruz Mother"
    assert m["stats"]["win_rate"] == 0.5
    assert m["exported_at"]


def test_explicit_gen_and_name(tmp_path):
    run = _run(tmp_path)
    dest = export_generation(tmp_path, run, gen=1, name="champ")
    assert dest.name == "champ"
    assert (dest / "model.zip").read_bytes() == b"w1"
    m = json.loads((dest / "manifest.json").read_text())
    assert m["gen"] == 1 and m["name"] == "champ"


def test_boss_defaults_to_hornet1_for_old_runs(tmp_path):
    run = _run(tmp_path, boss=None)
    m = json.loads(
        (export_generation(tmp_path, run) / "manifest.json").read_text())
    assert m["boss"] == "hornet1"
    assert m["boss_display"] == "Hornet Protector"


def test_name_collision_refused_unless_forced(tmp_path):
    run = _run(tmp_path)
    export_generation(tmp_path, run)
    with pytest.raises(ValueError, match="already exists"):
        export_generation(tmp_path, run)
    dest = export_generation(tmp_path, run, force=True)
    assert (dest / "model.zip").read_bytes() == b"w2"


def test_missing_explicit_gen_is_a_valueerror(tmp_path):
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="no complete checkpoint"):
        export_generation(tmp_path, run, gen=7)


def test_run_without_any_checkpoint_raises(tmp_path):
    run = tmp_path / "runs" / "empty"
    run.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        export_generation(tmp_path, run)


def test_exported_generations_reads_manifests_not_names(tmp_path):
    run = _run(tmp_path)
    export_generation(tmp_path, run, gen=1)
    export_generation(tmp_path, run, gen=2, name="renamed")  # name != r1_gen0002
    # Foreign and broken entries are skipped, never fatal.
    other = tmp_path / "exports" / "other"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"run_id": "r2", "gen": 5}))
    broken = tmp_path / "exports" / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("not json")
    assert exported_generations(tmp_path, "r1") == {1, 2}
    assert exported_generations(tmp_path, "r2") == {5}


def test_exported_generations_without_an_exports_dir_is_empty(tmp_path):
    assert exported_generations(tmp_path, "r1") == set()
