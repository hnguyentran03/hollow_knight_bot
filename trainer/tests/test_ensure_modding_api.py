import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "mod" / "ensure_modding_api.sh"

# Tiny stand-ins for the real assemblies: what matters is only whether the
# bytes contain the ModHooks marker the script greps for.
MODDED = b"\x00\x01MZ fake assembly ModHooks GetPlayerIntHook\x00"
VANILLA = b"\x00\x01MZ fake assembly PlayerData GameManager\x00"


def run_script(managed):
    return subprocess.run(
        ["bash", str(SCRIPT), str(managed)], capture_output=True, text=True)


def test_healthy_modded_dll_is_a_silent_noop(tmp_path):
    (tmp_path / "Assembly-CSharp.dll").write_bytes(MODDED)

    result = run_script(tmp_path)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    # Nothing created, nothing renamed.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Assembly-CSharp.dll"]


def test_vanilla_with_good_backup_swaps_and_keeps_vanilla_as_v(tmp_path):
    (tmp_path / "Assembly-CSharp.dll").write_bytes(VANILLA)
    (tmp_path / "Assembly-CSharp.dll.m").write_bytes(MODDED)

    result = run_script(tmp_path)

    assert result.returncode == 0
    assert "restored" in result.stdout.lower()
    assert (tmp_path / "Assembly-CSharp.dll").read_bytes() == MODDED
    assert (tmp_path / "Assembly-CSharp.dll.v").read_bytes() == VANILLA
    assert (tmp_path / "Assembly-CSharp.dll.m").read_bytes() == MODDED  # untouched


def test_vanilla_with_missing_backup_fails_loud_and_changes_nothing(tmp_path):
    (tmp_path / "Assembly-CSharp.dll").write_bytes(VANILLA)

    result = run_script(tmp_path)

    assert result.returncode != 0
    assert "lumafly" in result.stderr.lower()
    assert (tmp_path / "Assembly-CSharp.dll").read_bytes() == VANILLA
    assert not (tmp_path / "Assembly-CSharp.dll.v").exists()


def test_vanilla_with_vanilla_backup_fails_loud_and_changes_nothing(tmp_path):
    (tmp_path / "Assembly-CSharp.dll").write_bytes(VANILLA)
    (tmp_path / "Assembly-CSharp.dll.m").write_bytes(VANILLA)

    result = run_script(tmp_path)

    assert result.returncode != 0
    assert "lumafly" in result.stderr.lower()
    assert (tmp_path / "Assembly-CSharp.dll").read_bytes() == VANILLA
    assert not (tmp_path / "Assembly-CSharp.dll.v").exists()


def test_missing_active_dll_fails_loud(tmp_path):
    (tmp_path / "Assembly-CSharp.dll.m").write_bytes(MODDED)

    result = run_script(tmp_path)

    assert result.returncode != 0
    assert "lumafly" in result.stderr.lower()
    # The repair case must NOT fabricate an active dll from the backup.
    assert not (tmp_path / "Assembly-CSharp.dll").exists()
