# Multi-boss support — design

Date: 2026-08-02
Status: approved

## Goal

Make the boss a configuration axis instead of a hardcoded fact, and prove it by
training a policy to reliably beat a second boss. Validation target: **Gruz
Mother** (simplest suitable candidate: tiny moveset, low HP, flat arena, all
threats are her own body, single HealthManager). Hornet 1 remains the default;
existing runs and checkpoints keep working unchanged.

"Done" means a policy trained against Gruz Mother wins reliably, the way the
Hornet runs do — not just episodes running end-to-end.

## Non-goals

- Cross-boss policy transfer or a shared observation space. Each boss has its
  own FSM state list, so obs sizes differ and policies are boss-specific by
  construction.
- Multi-entity, phased, or fake-death bosses (Oro & Mato, Soul Master, etc.).
  The single-`HealthManager` `Die`-hook win detection is unchanged.
- Adding `target_kl` to the launcher whitelist (separate backlog item).

## Architecture: split registries, boss id over the wire

Chosen over (B) a trainer-owned registry that pushes config to the mod, and
(C) a mod-owned registry the trainer fetches at connect. The per-boss data
splits cleanly by consumer, so each side keeps only what it uses, keyed by a
shared boss id (`hornet1`, `gruz_mother`):

| Side | Owns per boss |
|---|---|
| Trainer (`trainer/hkrl/bosses.py`) | FSM state list (obs one-hot), arena constants (center X, half-width, floor Y, height) |
| Mod (registry in `EpisodeManager`) | boss scene name, statue-stand X in `GG_Workshop`, Attuned max-HP ceiling, LoadBoss tier index |

The reset macro already contains scene-specific behavior (the Hornet
scene-entry jump pulse), so per-boss C# branches are honest; a data-driven
"generic" mod would still need scene-conditional code. Cost accepted: adding a
boss touches both C# and Python and needs a mod rebuild.

## Trainer changes

- **`trainer/hkrl/bosses.py`** — a `BossSpec` dataclass (id, FSM state list,
  arena constants) and a registry. The Hornet constants at the top of `env.py`
  (`HORNET_STATES`, `ARENA_CENTER_X`, `ARENA_HALF_W`, `FLOOR_Y`,
  `ARENA_HEIGHT`) move into the `hornet1` spec.
- **`HKEnv`** takes a `BossSpec` and builds its observation space from it
  (scalar block + per-boss state one-hot). It sends the boss id in every reset
  request.
- **`train.py --boss`** — default `hornet1`, choices from the registry. The
  resolved value lands in `config.jsonl` like every other parameter.
- **Resume guard** — resuming a checkpoint whose recorded boss differs from
  the requested one is a hard error with a clear message (the obs shapes are
  incompatible; a shape crash is not an acceptable error surface). Boss is not
  resume-overridable, unlike `--target-kl`. Checkpoints from before this
  feature read as `hornet1`.

## Protocol changes

- The reset request gains `"boss": "<id>"`.
- The protocol version bumps, and the trainer refuses an old-version mod. An
  old mod would ignore the field and silently fight Hornet while the trainer
  builds a Gruz obs space — exactly the mismatch the version check exists to
  catch.
- A reset naming an id the mod doesn't know gets an error response, surfaced
  trainer-side as a clear failure (not a hang or a wrong fight).

## Mod changes

- Per-boss registry in `EpisodeManager`: id → scene, statue X, HP ceiling,
  tier index.
- The `BossScene` constant becomes the current episode's boss scene: win/loss
  and scene checks, the `sawSceneReentrySinceReset` latch, and the HP-ceiling
  tier verification all read from the active spec.
- The `GG_Workshop` macro walks to the per-boss statue X. The tier gate stays
  `LoadBoss(0)` for Attuned but is re-verified in-game per statue, not
  assumed.
- Hornet-specific macro behavior (scene-entry jump pulse) stays in a
  Hornet-only branch; Gruz gets her own branch only if discovery shows she
  needs entry handling.
- Win detection (`On.HealthManager.Die` hook) is unchanged in code but
  verified in-game against Gruz's burst-into-gruzzers death sequence.

## Dashboard changes

- Launch panel: boss dropdown, populated from the registry.
- Launcher parameter whitelist learns `boss`.
- Run cards and the active-run card display the boss from `config.jsonl`;
  runs without the field display as `hornet1`.

## In-game discovery (precedes training; user drives)

One session at the Gruz statue, recorded into new DISCOVERED.md sections and
then transcribed into the two registries:

1. FSM state names via the FSMLogger across several full fights.
2. Arena bounds off the F1 overlay (walls, floor, usable height).
3. Statue-stand X in `GG_Workshop`.
4. Attuned max-HP readings → the mod's ceiling value.
5. `LoadBoss(0)` tier verification at the Gruz statue.
6. A real kill reporting `result=WIN` (Die-hook verification).

## Error handling

- Unknown boss id: mod returns a protocol error; trainer fails the reset
  loudly.
- Mod/trainer version mismatch: trainer refuses at connect.
- Boss/checkpoint mismatch on resume: hard error before any training starts.
- Wrong difficulty tier: existing HP-ceiling abort, now per-boss.

## Testing

- `fake_game` learns boss ids: echoes the right scene per id, rejects unknown
  ids, so the protocol path is testable without the game.
- Env tests pin per-boss obs sizing from the spec.
- A resume-mismatch test pins the hard error.
- Launcher tests cover the `boss` whitelist entry.
- Existing tests keep passing with `hornet1` as the default.

## Rollout order

1. Registries + `--boss` flag + resume guard.
2. Protocol bump + mod registry/macro changes.
3. Tests green against `fake_game`.
4. In-game discovery session (fills the Gruz registry entries).
5. Smoke: a few real episodes end-to-end against Gruz (resets, wins, losses
   all reporting correctly).
6. Full training run to a reliably winning policy.
