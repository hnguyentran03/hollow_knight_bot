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
    # Proper name shown to humans (dashboard dropdown, nameplate, prompts):
    # the Hall of Gods statue name, which a prettify regex cannot derive
    # from the id ("hornet1" is "Hornet Protector", not "Hornet 1").
    display_name: str
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
        display_name="Hornet Protector",
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
        display_name="Gruz Mother",
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
    # Gorb (Hall of Gods, Attuned). States and arena measured 2026-08-05,
    # DISCOVERED.md section 8. Main FSM "Attacking".
    "gorb": BossSpec(
        id="gorb",
        display_name="Gorb",
        fsm_states=(
            "Init", "Wait", "Antic", "Attack", "Recover", "Damaged",
            "Double Pause", "Anim", "Triple Pause",
            "UNKNOWN",
        ),
        arena_center_x=56.0,
        arena_half_w=11.87,
        floor_y=33.40,
        arena_height=10.84,
    ),
    # Soul Warrior (Hall of Gods, Attuned). States and arena measured
    # 2026-08-05, DISCOVERED.md section 9 (walls 35.01/58.94, floor 5.39,
    # top 19.09). Main FSM "Mage Knight".
    "soul_warrior": BossSpec(
        id="soul_warrior",
        display_name="Soul Warrior",
        fsm_states=(
            "GG Pause", "Up Tele", "Stomp Antic", "Stomp Air",
            "Stomp Recover", "Idle", "Slash Antic", "Dash", "Slash Recover",
            "Tele Antic", "Side Tele", "Shoot Antic", "Shoot", "Shoot CD",
            "Slash", "Televade", "Evade", "Stomp Slash", "Evade Antic",
            "Evade Recover",
            "UNKNOWN",
        ),
        arena_center_x=46.97,
        arena_half_w=11.96,
        floor_y=5.39,
        arena_height=13.70,
    ),
    # Marmu (Hall of Gods, Attuned). States and arena measured 2026-08-05,
    # DISCOVERED.md section 10 (walls 51.27/88.73, floor 10.40, top 23.09).
    # Main FSM "Control".
    "marmu": BossSpec(
        id="marmu",
        display_name="Marmu",
        fsm_states=(
            "Start Pause", "Antic", "Chase", "Unroll", "Warp Out 2",
            "UNKNOWN",
        ),
        arena_center_x=70.0,
        arena_half_w=18.73,
        floor_y=10.40,
        arena_height=12.69,
    ),
    # False Knight (Hall of Gods, Attuned). States and arena measured
    # 2026-08-05, DISCOVERED.md section 11 (walls 11.19/45.70, floor 27.40,
    # top 42.81). Main FSM "FalseyControl".
    "false_knight": BossSpec(
        id="false_knight",
        display_name="False Knight",
        fsm_states=(
            "Start Fall", "State 1", "First Idle", "Jump Antic", "Rise",
            "Fall", "Idle", "JA Antic", "JA Rise", "JA Fall", "JA Hit",
            "JA Recoil 2", "Turn R", "S Attack Antic", "S Attack Recover",
            "Run Antic", "Run", "JA Recoil", "JA End", "S Antic", "S Rise",
            "S Fall", "S Land", "Stun In Air", "Pause Short", "Open Uuup",
            "Opened", "Hit", "Recover", "Idle Pause", "Rage Jump Antic",
            "Rise 2", "Fall 2", "State 2", "R Attack Antic", "Rage",
            "Particle Pause", "Anim End", "Stun Land", "Rage End",
            "Death Open", "Opened 2", "Hit 2", "Death Anim Start", "Steam",
            "Ready", "Blow", "Death Head Land", "Cough", "S Attack", "Slam",
            "Stun Fail", "Turn L", "JA Slam",
            "UNKNOWN",
        ),
        arena_center_x=28.45,
        arena_half_w=17.26,
        floor_y=27.40,
        arena_height=15.41,
    ),
}

# The boss used when nothing specifies one -- the original fight, kept as
# the default so pre-multi-boss run configs and muscle memory keep working.
# A plain id string, not a BossSpec: every consumer wants the id.
DEFAULT_BOSS = "hornet1"


def get_boss(boss_id: str) -> BossSpec:
    try:
        return BOSSES[boss_id]
    except KeyError:
        known = ", ".join(sorted(BOSSES))
        raise ValueError(
            f"unknown boss {boss_id!r}; known bosses: {known}") from None
