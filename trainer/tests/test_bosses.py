import pytest

from hkrl.bosses import BOSSES, DEFAULT_BOSS, get_boss


def test_hornet1_spec_matches_the_measured_constants():
    # The values recorded in mod/DISCOVERED.md sections 1-2; moving them into
    # the registry must not change them.
    spec = get_boss("hornet1")
    assert spec.id == "hornet1"
    assert len(spec.fsm_states) == 38          # 37 recorded states + UNKNOWN (10 surfaced 2026-08-07 by the unseen-state warning)
    assert spec.fsm_states[-1] == "UNKNOWN"
    assert spec.arena_center_x == 26.5
    assert spec.arena_half_w == 11.23
    assert spec.floor_y == 28.41
    assert spec.arena_height == 9.59


def test_get_boss_rejects_unknown_ids_naming_the_known_ones():
    with pytest.raises(ValueError, match="hornet1"):
        get_boss("grimm")


def test_registry_keys_match_spec_ids():
    assert all(spec.id == key for key, spec in BOSSES.items())


def test_gruz_mother_is_registered_with_its_own_obs_space():
    spec = get_boss("gruz_mother")
    assert spec.id == "gruz_mother"
    assert spec.fsm_states[-1] == "UNKNOWN"
    # Different state list -> different obs size -> boss-specific policies.
    assert spec.fsm_states != get_boss("hornet1").fsm_states


def test_gorb_is_registered_with_its_own_obs_space():
    spec = get_boss("gorb")
    assert spec.id == "gorb"
    assert spec.fsm_states[-1] == "UNKNOWN"
    assert spec.fsm_states != get_boss("hornet1").fsm_states


def test_soul_warrior_is_registered_with_its_own_obs_space():
    spec = get_boss("soul_warrior")
    assert spec.id == "soul_warrior"
    assert spec.fsm_states[-1] == "UNKNOWN"
    assert spec.fsm_states != get_boss("hornet1").fsm_states


def test_marmu_is_registered_with_its_own_obs_space():
    spec = get_boss("marmu")
    assert spec.id == "marmu"
    assert spec.fsm_states[-1] == "UNKNOWN"
    assert spec.fsm_states != get_boss("hornet1").fsm_states


def test_false_knight_is_registered_with_its_own_obs_space():
    spec = get_boss("false_knight")
    assert spec.id == "false_knight"
    assert spec.fsm_states[-1] == "UNKNOWN"
    assert spec.fsm_states != get_boss("hornet1").fsm_states


def test_every_boss_has_a_display_name():
    # display_name is required and human-facing: non-empty, no underscores,
    # not just the id echoed back.
    for spec in BOSSES.values():
        assert spec.display_name.strip()
        assert "_" not in spec.display_name


def test_display_names_use_the_hall_of_gods_statue_names():
    # hornet1 is the id the regex can never get right; the others pin the
    # statue names.
    assert get_boss("hornet1").display_name == "Hornet Protector"
    assert get_boss("soul_warrior").display_name == "Soul Warrior"
    assert get_boss("false_knight").display_name == "False Knight"


def test_default_boss_is_a_registered_id():
    assert DEFAULT_BOSS in BOSSES
