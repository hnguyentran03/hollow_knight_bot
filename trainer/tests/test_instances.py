from pathlib import Path
from hkrl.instances import (
    UNITY_SAVE_SUBPATH, instance_home, port_for, provision,
)


def _seed(tmp_path):
    src = tmp_path / "seed"
    src.mkdir()
    (src / "user1_1.5.12620.dat").write_text("save")
    (src / "shared.dat").write_text("shared")
    return src


def test_port_for_offsets_from_base():
    assert port_for(0) == 9020
    assert port_for(3) == 9023
    assert port_for(1, base_port=9100) == 9101


def test_provision_seeds_the_unity_save_directory(tmp_path):
    home = provision(2, root=tmp_path, seed_from=_seed(tmp_path))
    assert home == instance_home(2, tmp_path)
    saves = home / UNITY_SAVE_SUBPATH
    assert (saves / "user1_1.5.12620.dat").read_text() == "save"
    assert (saves / "shared.dat").read_text() == "shared"


def test_provision_is_idempotent_and_preserves_progress(tmp_path):
    seed = _seed(tmp_path)
    home = provision(0, root=tmp_path, seed_from=seed)
    saves = home / UNITY_SAVE_SUBPATH
    (saves / "user1_1.5.12620.dat").write_text("progressed")

    provision(0, root=tmp_path, seed_from=seed)

    # Re-provisioning must not clobber a save that has since diverged --
    # relaunching instances between training runs is routine.
    assert (saves / "user1_1.5.12620.dat").read_text() == "progressed"
