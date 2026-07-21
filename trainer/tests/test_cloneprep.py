import pytest

from hkrl import cloneprep


def test_strip_foreign_profiles_keeps_slot_1_deletes_others(tmp_path):
    for name in [
        "user1.dat", "user1_1.5.78.11833.dat", "user1.modded.json",
        "user1.dat.bak756",
        "user4.dat", "user4_1.5.78.11833.dat", "user4.modded.json",
        "user2.dat",
        "shared.dat", "ModLog.txt", "AppConfig.ini",
    ]:
        (tmp_path / name).write_text("x")
    # A directory whose name matches the profile pattern must be skipped,
    # not unlinked (unlink on a dir raises).
    (tmp_path / "user9.backups").mkdir()

    # The private primitive is exercised directly here; production callers
    # can only reach it through prepare_clone_save's clone-name guard.
    removed = cloneprep._strip_foreign_profiles(tmp_path)

    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert survivors == [
        "AppConfig.ini", "ModLog.txt", "shared.dat",
        "user1.dat", "user1.dat.bak756", "user1.modded.json",
        "user1_1.5.78.11833.dat", "user9.backups",
    ]
    assert sorted(p.name for p in removed) == [
        "user2.dat", "user4.dat", "user4.modded.json",
        "user4_1.5.78.11833.dat",
    ]


def test_prepare_clone_save_strips_only_the_foreign_profiles(tmp_path):
    # A clone save dir is named "<master bundle id>.hkrl<port>".
    clone = tmp_path / "unity.Team Cherry.Hollow Knight.hkrl9020"
    clone.mkdir()
    for name in ("user1.dat", "user4.dat", "user4.modded.json", "shared.dat"):
        (clone / name).write_text("x")

    cloneprep.prepare_clone_save(clone)

    assert sorted(p.name for p in clone.iterdir()) == ["shared.dat", "user1.dat"]


def test_prepare_clone_save_refuses_a_non_clone_dir(tmp_path):
    # The master save dir (no .hkrl<port> suffix) must be refused outright, so a
    # caller bug can never delete real profiles. Data-safety backstop.
    master = tmp_path / "unity.Team Cherry.Hollow Knight"
    master.mkdir()
    (master / "user1.dat").write_text("x")
    (master / "user4.dat").write_text("x")
    with pytest.raises(ValueError):
        cloneprep.prepare_clone_save(master)
    assert (master / "user4.dat").exists()  # untouched
