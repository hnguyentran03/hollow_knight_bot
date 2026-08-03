"""Per-boss trainer-side data: FSM state lists and arena constants.

One BossSpec per supported boss, keyed by a boss id shared with the mod's
BossRegistry (mod/BossRegistry.cs). The trainer sends the id in every reset
request; each side keeps only the data it consumes, so adding a boss means
one entry here (obs-space data) and one in the mod (scene/statue/ceiling
data), both transcribed from an in-game discovery session recorded in
mod/DISCOVERED.md.

The FSM state list sizes the observation one-hot, so policies and
checkpoints are boss-specific by construction: two bosses with different
state lists have incompatible observation spaces.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BossSpec:
    id: str
    # FSM state names recorded from live play, ending with the "UNKNOWN"
    # fallback slot every unseen state maps to.
    fsm_states: tuple[str, ...]
    # Arena normalization constants, measured off the F1 overlay (see
    # mod/DISCOVERED.md): horizontal is center-relative (x - center) / half_w,
    # vertical is floor-relative (y - floor) / height.
    arena_center_x: float
    arena_half_w: float
    floor_y: float
    arena_height: float


BOSSES = {
    # Hornet 1 (Hall of Gods, Attuned). States: DISCOVERED.md section 1;
    # arena: section 2 (walls 15.27/37.73, floor 28.41, top 38).
    "hornet1": BossSpec(
        id="hornet1",
        fsm_states=(
            "Flourish", "Run", "A Dash", "Hard Land", "Idle", "Throw Antic",
            "Thrown", "Throw Recover", "In Air", "ADash Antic", "Run Antic",
            "G Dash Antic", "G Dash", "Jump Antic", "Land", "GDash Recover1",
            "GDash Recover2", "Evade", "Evade Antic", "Evade Land", "Wall L",
            "Sphere A", "Sphere Antic A", "Sphere Recover A", "Wall R",
            "Stun Air", "Stun Land",
            "UNKNOWN",
        ),
        arena_center_x=26.5,
        arena_half_w=11.23,
        floor_y=28.41,
        arena_height=9.59,
    ),
    # Gruz Mother (Hall of Gods, Attuned). States and arena measured
    # 2026-08-03, DISCOVERED.md sections 6 and 7 (walls 86.27/102.73,
    # floor 15.40, top 24.66). Main FSM "Big Fly Control".
    "gruz_mother": BossSpec(
        id="gruz_mother",
        fsm_states=(
            "Wake", "GG Extra Pause", "Buzz", "Charge Antic", "Charge",
            "Charge Recover D", "Super End", "Slam Antic", "Flying",
            "Slam Up", "Slam Down", "Launch Down", "Slam End",
            "Charge Recover U", "Charge Recover L", "Launch Up",
            # Surfaced by the trainer's unseen-state warning during the
            # 2026-08-03 smoke run; discovery never saw a right-wall recover.
            "Charge Recover R",
            "UNKNOWN",
        ),
        arena_center_x=94.5,
        arena_half_w=8.23,
        floor_y=15.4,
        arena_height=9.26,
    ),
}


def get_boss(boss_id: str) -> BossSpec:
    try:
        return BOSSES[boss_id]
    except KeyError:
        known = ", ".join(sorted(BOSSES))
        raise ValueError(
            f"unknown boss {boss_id!r}; known bosses: {known}") from None
