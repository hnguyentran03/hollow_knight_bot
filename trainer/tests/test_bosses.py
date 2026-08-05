import pytest

from hkrl.bosses import BOSSES, get_boss


def test_hornet1_spec_matches_the_measured_constants():
    # The values recorded in mod/DISCOVERED.md sections 1-2; moving them into
    # the registry must not change them.
    spec = get_boss("hornet1")
    assert spec.id == "hornet1"
    assert len(spec.fsm_states) == 28          # 27 recorded states + UNKNOWN
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
