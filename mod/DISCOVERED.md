# DISCOVERED.md — values recorded from live play with the echo overlay

This file records ground-truth numbers read off the on-screen overlay (Task 2,
`OverlayUI.cs`, toggled with **F1**) while actually playing the game.
**Recorded in a live play session on 2026-07-18. All slots are filled.**

Do not change a value unless you personally read the new one off the overlay
while playing. Do not estimate, guess, or infer a "plausible" number — later
tasks (the Python trainer's `HORNET_STATES` list in Task 7, the arena bounds
used for normalizing observations, and `StatueX` in Task 6) consume these
values verbatim and a wrong-but-plausible number here will silently corrupt
the bot's observations with no visible error.

These numbers are specific to the game build measured against. If Steam
updates the game, re-verify them rather than assuming they carried over.

## How to gather these values

1. Run `./mod/build.sh` from the repo root (rebuilds and installs the mod).
2. Launch Hollow Knight. Load the Godhome save.
3. Go to Hall of Gods and start **Hornet 1 (Attuned)**.
4. The overlay is on by default; press **F1** to toggle it off/on if it's in
   the way.
5. Read the live `Knight:` and `Boss:` lines described below and record the
   numbers here.

---

## 1. Hornet FSM state names

Fight Hornet 1 for a few minutes (a few attempts is fine — you don't need to
win) and watch the `state=` field on the overlay's `Boss:` line. Write down
every distinct string you see. Expect something like Idle/Run/Evade/Throw/
Throw Antic/A Dash/G Dash/Sphere/Jump/Land/Stun, but **record exactly what the
overlay prints**, not what you expect to see — do not copy the example list
below into the final answer without having verified it against the overlay.

```
Flourish, Run, A Dash, Hard Land, Idle, Throw Antic, Thrown, Throw Recover, In Air, ADash Antic, Run Antic, G Dash Antic, G Dash, Jump Antic, Land, GDash Recover1, GDash Recover2, Evade, Evade Antic, Evade Land, Wall L, Sphere A, Sphere Antic A, Sphere Recover A, Wall R, Stun Air, Stun Land
```

## 2. Arena bounds (Hall of Gods, Hornet 1 arena)

Walk the Knight to the left wall of the arena and note the `x=` value on the
`Knight:` line. Do the same at the right wall. Also note the `y=` value while
standing on the arena floor (not jumping/falling).

```
Knight X at left wall:   15.27
Knight X at right wall:  37.73
Floor Y:                 28.41
```

Derived values (compute only after both X values above are filled in):

```
Arena center X     = (left + right) / 2   = 26.5
Arena half-width X = (right - left) / 2   = 11.23
```

For the vertical scale, measure the **top of the arena** the same way the
walls above gave the horizontal scale: jump as high as possible (full jump +
double jump) to reach the ceiling, press up into it, and read the steady `y=`.
If the ceiling is genuinely out of reach, record the `y` at the top of the
two-jump apex instead. Using the arena top (not a Knight-specific apex) keeps
this consistent with the wall-based horizontal scale and bounds Hornet's
aerial leaps too, so it covers the boss `y` as well. Confirm `y` INCREASES
going up (top > floor); if your build reads the top as smaller than the floor,
y is inverted and the height is `floor - top` instead.

```
Knight Y at arena top (ceiling, or two-jump apex if unreachable): 38
```

Derived value (compute only after the top above is filled in):

```
Arena half-height H = top - floor = 9.59
```

This feeds `ARENA_HEIGHT` in `trainer/hkrl/env.py` (= 9.59). No `/2`, unlike
the horizontal `ARENA_HALF_W`: vertical is normalized floor-relative,
`(ky - FLOOR_Y) / ARENA_HEIGHT`, so the floor maps to 0 and the top to ~1 --
it's the full floor-to-top height, not a half-span from a center.

## 3. Statue-stand X in `GG_Workshop`

In `GG_Workshop` (the Godhome hub room), stand at the Hornet statue and note
the Knight's `x=` value from the overlay.

```
Knight X at Hornet statue in GG_Workshop: 62.21
```

---

## Status

- [x] Hornet FSM state list recorded (27 distinct states)
- [x] Arena X range recorded (both walls)
- [x] Floor Y recorded
- [x] Arena center / half-width computed
- [x] Statue X recorded in GG_Workshop
- [x] Arena vertical scale (arena top Y = 38, height 9.59) recorded —
      `ARENA_HEIGHT` in `trainer/hkrl/env.py`

**Complete.** The arena top Y was measured in a later session and feeds
`ARENA_HEIGHT = 9.59` in `trainer/hkrl/env.py`; all other values are from the
2026-07-18 session.

---

## 4. Statue challenge menu: live selected-tier signal

**Recorded in a live play session on 2026-07-21**, using a temporary
diagnostic in `mod/EpisodeManager.cs` (a `DiscoverStatueTier`-gated block at
the top of `LateUpdate`, since removed once this was confirmed) that logged
candidate signals every 2s while a human opened the Hornet statue's challenge
menu and cycled the difficulty across Attuned/Ascended/Radiant.

**Live signal (used):** the Unity EventSystem's current selection. With the
challenge menu open,

```
EventSystem.current.currentSelectedGameObject
```

equals one of `ui.tier1Button.button.gameObject` / `ui.tier2Button.button
.gameObject` / `ui.tier3Button.button.gameObject` (`ui =
FindObjectOfType<BossChallengeUI>()`), and tracks the highlighted tier LIVE
as the human cycles it -- verified 0->1->2->0 while cycling, and only ever
0/1/2 while the menu was open. `FindObjectOfType<BossChallengeUI>() != null`
is what carries the "-1 means unreadable" contract: with the menu closed
there is no `BossChallengeUI` instance to read from at all.

**Rejected candidate 1:** `BossChallengeUI`'s static `lastSelectedButton`
field. Read via reflection alongside the EventSystem signal in the same
diagnostic session; it only updates on a selection *event*, not live while
cycling -- it lagged behind the EventSystem reading and does not track the
highlight in real time. Not usable as the per-cycle read.

**Rejected candidate 2 (as predicted by the original spec):**
`PlayerData.instance.bossStatueTargetLevel`. Logged every 2s through the
entire cycling session; it stayed at a global `-1` the whole time and never
moved pre-confirm, confirming the spec's prediction (R3.4) and the save dig's
earlier `-1` reading. The game only writes this field on CONFIRM (the level
load), not on highlight change.

**Correction (used):** writing the selection directly,

```
EventSystem.current.SetSelectedGameObject(ui.tier1Button.button.gameObject)
```

selects Attuned. Because the read side above proved the EventSystem
selection IS the highlight (not merely correlated with it), writing it is a
direct, exact correction -- no synthetic tier-navigation input (e.g. an Up/
Down press toward Attuned) is needed.

Implemented in `mod/StateReader.cs` as `ReadSelectedChallengeTier()` and
`SelectAttunedChallengeTier()`, and wired into the statue-menu branch of
`ResetMacro.Tick()` in `mod/EpisodeManager.cs`: the confirm pulse only fires
when the tier reads `0` (Attuned); a nonzero tier triggers the correction
instead of a confirm that cycle.

**Trainer-context correction (verified in a real N=1 training run,
2026-07-21):** the `-1` reading is NOT reliably "menu not open yet, or a
broken signal" -- it also covers "menu open, nothing selected", and in a
trainer-driven (unfocused/automated) game window that second case is the
NORMAL one, not an edge case. A live run showed `ReadSelectedChallengeTier()`
returning `-1` through the ENTIRE statue-menu phase of every reset,
including the cycle right before the confirm landed on an open menu -- so
the original gate (which let the existing confirm timing proceed unchanged
on any `-1`) never actually verified a tier in that run; every fight entry
was the blind-confirm fallback. A manual, focused session did not show this:
the reader read live `0`/`1`/`2` as a human cycled tiers. Root cause: with
the game window unfocused/automated, the menu's select-on-open never fires
(Unity's EventSystem has no `currentSelectedGameObject` when the menu
appears), so "menu open but unselected" and "menu not open" collapse into
the same `-1`.

**Fix (superseded below by the LoadBoss(0) entry point):** `StateReader.cs`
adds `IsChallengeMenuOpen()` (`FindObjectOfType<BossChallengeUI>() != null`,
exception-safe), letting `EpisodeManager`'s gate split the two cases the old
code couldn't tell apart:
- menu not open: proceed unchanged, no log (this fires on every normal
  pre-open window and previously spammed a "gate inactive" line for no
  reason).
- menu open, tier `0`: allow the confirm (silent, healthy path).
- menu open, tier `>0` or `-1`: call `SelectAttunedChallengeTier()` and veto
  the confirm.

`SetSelectedGameObject` is focus-independent (it writes engine state
directly rather than relying on input-driven UI navigation), which is why
the mod can select Attuned itself even though the game never does so on its
own in this environment. **However** -- see the next section -- a second
real run showed even this write-side call does not reliably stick when the
window is unfocused, which is why the gate now falls through to
`LoadBoss(0)` rather than retrying `SelectAttunedChallengeTier()`
indefinitely.

---

## 5. Unfocused EventSystem selection is refused outright; `LoadBoss(int)` entry

**Verified in a second real N=1 trainer run (2026-07-21, game unfocused).**
Section 4's fix above assumed `SetSelectedGameObject` would land the
Attuned correction even when the game window is unfocused/automated (it
writes engine state directly, not through input navigation, so this seemed
focus-independent). That assumption did not hold: the gate's
`SelectAttunedChallengeTier()` fired twice ("attempt 1/2", "attempt 2/2" in
the old 2-attempt scheme) and the very next
`ReadSelectedChallengeTier()` re-read stayed `-1` both times -- the write
never stuck. So *writing* the EventSystem selection is refused unfocused,
the same as the engine never producing one on its own (section 4). The old
fallback for this case was a blind confirm (press Jump with no tier
verification), which enters whatever tier the menu happens to default to --
not a real fix, just a documented last resort.

**Entry point that bypasses the EventSystem/focus dependency entirely:**
the disassembled `BossChallengeUI` has `LoadBoss(int level)` (and an
overload `LoadBoss(int level, bool doHideAnim)`) -- the method the tier
buttons' own `OnClick` invokes. `level 0` = Attuned. Calling it requests the
tier BY VALUE: no highlight, no `EventSystem.current`, no window-focus
dependency of any kind.

Implemented in `mod/StateReader.cs` as `ConfirmAttunedChallenge()`: finds
the open `BossChallengeUI` (null -> `false`), then invokes `LoadBoss` via
reflection --

```
ui.GetType().GetMethod("LoadBoss",
    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
    null, new[] { typeof(int) }, null)
    .Invoke(ui, new object[] { 0 })
```

-- rather than a direct call, because `LoadBoss`'s member visibility on the
disassembled type is unverified and there are two overloads; reflection
with an explicit `int`-only parameter filter finds the right one regardless
of visibility. Returns `true` on success, `false` on any failure (null UI,
missing method, or an exception from inside `LoadBoss` itself) --
exception-safe like this file's other helpers.

**Updated decision table** (`EpisodeManager.cs`'s `ResetMacro.Tick()`
statue-menu branch at the time; hoisted into `ResetMacro.GateConfirm()` and
generalized to run for any `GG_Workshop` branch -- see the budget-correction
entry below):
- menu not open: proceed unchanged, no log (silent, unchanged from
  section 4).
- menu open, tier `0` (Attuned): allow the confirm pulse (silent, healthy
  path, unchanged).
- menu open, tier `-1` or `>0`: **one** `SelectAttunedChallengeTier()`
  attempt (reduced from 2 -- selection is now known-refused unfocused, so a
  second attempt only costs time; one attempt still lets a focused/manual
  session, where the correction actually sticks, resolve here), veto the
  confirm, log the attempt including the selection call's `bool` return.
- still not Attuned the next cycle: call `ConfirmAttunedChallenge()`
  (`LoadBoss(0)`). On `true`: log "entering Attuned directly via
  LoadBoss(0)", veto the manual confirm pulse (the scene change into
  `GG_Hornet_1` follows on its own), and latch so the gate does not
  re-invoke `LoadBoss` every subsequent cycle this visit. The latch resets
  whenever the menu is not open (mirroring `tierGateAttempts`), so every
  fresh statue visit gets its own attempt.
- only if `ConfirmAttunedChallenge()` itself returns `false` (null UI,
  reflection miss, or an exception inside `LoadBoss`): fall back to the
  pre-existing loud blind-confirm log line and blind confirm -- the gate
  never withholds the confirm indefinitely, since starving the macro turns
  a readability problem into a dead run (budget expiry -> `InstanceDown`).
  Backstop B (`EpisodeManager.TickReset`'s `fightLive` HP-ceiling check)
  still guards against a wrong tier slipping through even here.

**Budget correction (2026-07-21, corrects the entry above from commit
`0413062`):** that commit raised `ResetMacroBudgetSeconds` from 22.5s to 40s
in response to the cold-boot measurement above (title screen to statue-menu
at ~20.4s of the then-22.5s budget, cold boots measured ~24-26s end-to-end).
This was reverted back to 22.5s: the budget is a deliberate INVARIANT, not a
timing knob to raise past a slow cold boot. It must stay strictly under the
Python trainer's 30s socket timeout (`Connection(timeout=30.0)` in
`trainer/hkrl/env.py`) -- at or above 30s, the trainer's own `reset()` call
times out and tears down the connection before the mod's budget-expiry check
ever runs, which trades the mod's own lightweight "log where it got stuck,
drop, let the trainer reconnect" path for the trainer's heavyweight
wedged-instance recovery (full supervisor relaunch) on every cold boot --
exactly the failure the design below avoids. A cold boot-to-fight
legitimately runs longer than one 22.5s budget; `trainer/hkrl/env.py`
documents this as intentional: a dropped/expired reset is retried by
reconnecting, and because menu/scene progress (title screen -> save select
-> stood up at the bench -> walk-to-statue -> challenge menu) persists
across drops, successive resets ratchet forward -- each one picking up
further along than the last -- until the fight is live. A cold boot is
therefore expected to span SEVERAL 22.5s budgets, never one long one. Mid-run
resets (~9s total) fit comfortably inside a single budget either way.

**The real gap the ratchet exposed (also verified in-game 2026-07-21):** the
40s budget masked, but did not cause, a second bug in the ratchet itself.
On the ratchet's second (or later) attempt, `ResetMacro.Reset()` zeroes
`statueMenuLatched`, so the fresh attempt re-derives its branch purely from
knight position and restarts at `workshop-settling` -> `step-off-statue` --
but the PRIOR attempt's challenge menu can still be open on screen (its
budget expired, or its connection dropped, before the resulting scene change
was ever observed). `step-off-statue`'s own `step-off-menu-recover`
stall-clear pulsed a **blind** `Jump` confirm into that leftover menu -- no
tier read, no gate -- bypassing the tier gate entirely. Observed twice: a
scene change straight into `GG_Hornet_1` during `step-off-statue`, the
`statue-menu` branch never reached that reset at all.

**Fix:** the tier gate's decision table (unchanged itself -- read tier, one
`SelectAttunedChallengeTier()` attempt logged, `ConfirmAttunedChallenge()`
latched, blind-confirm last resort) is hoisted out of the `statue-menu`
branch into `ResetMacro.GateConfirm()` (`mod/EpisodeManager.cs`) and invoked
from ONE site at the end of the `GG_Workshop` scene handling in `Tick()`,
after branch selection, on every tick where `StateReader
.IsChallengeMenuOpen()` reads true -- regardless of which branch
(`workshop-settling`, `step-off-statue`, `walk-to-statue`, `statue-menu`)
position-based selection landed on. Its return value overrides `b.Jump` --
the confirm-capable input -- for that tick; branch-generated movement inputs
(`Left`/`Right`/`Up`) are computed separately and are left alone. When the
menu is not open, the gate is not invoked and every branch behaves exactly
as before the hoist; the `statue-menu` branch's own Up-press-then-blind-pulse
behavior before the menu is detected open is preserved as the special case
it becomes (it is the one branch that itself presses Up to open the menu).
`tierGateAttempts`/`attunedConfirmedViaLoadBoss`/`blindConfirmFallbackLogged`
now reset whenever `IsChallengeMenuOpen()` reads false on ANY `GG_Workshop`
tick (previously only checked inside the `statue-menu` branch), matching
their existing "zeroed every tick the menu is not open" contract.

## 6. Gruz Mother FSM state names

Recorded 2026-08-03 with the F4 `DiscoveryLogger` across two sessions (the
first session captured no states — the logger watched only an FSM named
`Control`, Hornet's name; fixed to log every FSM on a HealthManager owner,
tagged with the FSM name). Boss GameObject: `Giant Fly`. Main FSM:
**`Big Fly Control`**, 16 distinct states in first-seen order over full
Attuned fights:

```
Wake, GG Extra Pause, Buzz, Charge Antic, Charge, Charge Recover D,
Super End, Slam Antic, Flying, Slam Up, Slam Down, Launch Down, Slam End,
Charge Recover U, Charge Recover L, Launch Up
```

A secondary `bouncer_control` FSM on the same object (`Stopped`, `Fly 2`)
is movement plumbing and is not transcribed. Because the main FSM's name is
boss-specific, `BossRegistry.BossSpec` gained an `FsmName` field and
`StateReader` locates the boss FSM through it.

## 7. Gruz Mother arena, statue, and HP

Same 2026-08-03 sessions, via the DiscoveryLogger's knight-extreme and
statue lines (walls tagged, ceiling jumped mid-fight):

```
Scene (Attuned):         GG_Gruz_Mother   (Ascended: GG_Gruz_Mother_V)
Knight X at left wall:   86.27
Knight X at right wall:  102.73
Floor Y:                 15.40
Knight Y at arena top:   24.66
Arena center X    = (86.27 + 102.73) / 2 = 94.50
Arena half-width  = (102.73 - 86.27) / 2 = 8.23
Arena height      = 24.66 - 15.40        = 9.26
Knight X at Gruz statue in GG_Workshop: 28.0 (settled readings 27.96-28.08)
Max HP: 650 Attuned, 945 Ascended -> mod ceiling MaxAttunedHp = 700
```

`LoadBoss(0)` was not separately re-verified at the Gruz statue; the 700
HP ceiling (backstop B) catches a wrong tier during the smoke run, and the
Die-hook win report through the burst-into-gruzzers death sequence is also
verified there.

**Addendum (2026-08-03 smoke):** the trainer's unseen-state warning surfaced
`Charge Recover R` during the first live gruz run -- the discovery fights
never saw a right-wall charge recovery. Added to the `gruz_mother`
`fsm_states` (17 recorded states + UNKNOWN).

## 8. Gorb FSM states, arena, statue, and HP

Recorded 2026-08-05 with the F4 `DiscoveryLogger`, plus per-visit analysis of
the raw ModLog, plus two user-supplied edge readings (arena has no side
walls -- see below). Scene (Attuned): `GG_Ghost_Gorb` (Ascended:
`GG_Ghost_Gorb_V`). Boss GameObject: `Ghost Warrior Slug`. Main FSM:
**`Attacking`**, 9 distinct states in first-seen order, the FSM that cycles
attack-like states (Antic/Attack/Recover) per the parser guidance:

```
Init, Wait, Antic, Attack, Recover, Damaged, Double Pause, Anim, Triple Pause
```

A `Movement` FSM on the same object (`Warp In`, `Hover`, `Attacking`,
`Warp Check`) is warp/hover plumbing and is not transcribed (same precedent
as Gruz's untranscribed `bouncer_control`). Other minor FSMs seen and not
transcribed: `Distance Attack` (`Close`/`Away`), `Warp messenger` (`Wait`),
`Broadcast Ghost Death` (`Idle`).

**Arena edge-override provenance:** the arena has no side walls -- the floor
ends in lethal drops -- so the DiscoveryLogger's knight-X extremes include
falls past the edges (per-visit final ranges 42.44-68.78, 32.29-79.38,
42.61-69.70), which are not usable as wall readings. Instead the user read
the pre-drop edge X values directly off the overlay and supplied them:

```
Left edge X (overlay read):  44.14
Right edge X (overlay read): 67.87
Floor Y (all three Attuned visits, settled): 33.40
Arena center X    = (44.14 + 67.87) / 2 = 56.0    (56.005)
Arena half-width  = (67.87 - 44.14) / 2 = 11.87   (11.865)
```

**Arena top / height:** the per-visit ceiling-press tops were 38.51, 43.16,
and 44.24; the highest across visits, 44.24, is used as the arena top
(consistent with the Hornet/Gruz convention of taking the highest measured
top), giving:

```
Arena top (highest across visits): 44.24
Arena height = 44.24 - 33.40 = 10.84
```

The lower per-visit tops (38.51, 43.16) corroborate a height in the same
ballpark (~10) rather than indicating a measurement error.

**Statue:** settled menu-open readings 126.24, 126.25, 126.22, 126.23
(first-approach outlier 134.66 excluded). `StatueX = 126.23` -- the macro's
+/-0.5 settle window (125.73-126.73) contains all settled readings (Gruz
28.0->28.6 lesson satisfied, so no post-hoc nudge is needed here).

**HP:** Max HP 650 Attuned (three visits, consistent), 1000 Ascended
(`GG_Ghost_Gorb_V`) -> mod ceiling `MaxAttunedHp = 700` (same margin style as
Gruz's 650/945 -> 700). `TierIndex = 0`, re-verify at this statue during
smoke (registry comment convention).

**Projectile:** `NeedleName = null`. `Shot Slug Spear(Clone)` was logged
with 96 distinct instance ids in `GG_Ghost_Gorb` -- per-shot clones, not
trackable by the mod's find-by-name mechanism. `Spike Collider` /
`Spike Collider (1)` (3 and 3 instances) are arena hazards, not a boss
projectile.

## 9. Soul Warrior FSM states, arena, statue, and HP

Recorded 2026-08-05 with the F4 `DiscoveryLogger`, plus per-visit analysis of
the raw ModLog. Scene (Attuned): `GG_Mage_Knight` (Ascended:
`GG_Mage_Knight_V`). Boss GameObject: `Mage Knight`. Main FSM (named after
the object): **`Mage Knight`**, 20 distinct states in first-seen order:

```
GG Pause, Up Tele, Stomp Antic, Stomp Air, Stomp Recover, Idle, Slash Antic,
Dash, Slash Recover, Tele Antic, Side Tele, Shoot Antic, Shoot, Shoot CD,
Slash, Televade, Evade, Stomp Slash, Evade Antic, Evade Recover
```

In the Ascended fight only, four `Mage Balloon Spawner (N)` objects (hp 13
each, FSM `Control`: `Spawn` / `Chase - In Sight`) appeared -- Ascended-only
summons, not part of the Attuned fight, and not transcribed (same precedent
as Gruz's untranscribed `bouncer_control` and Gorb's untranscribed
`Movement`).

**Arena:** two independent Attuned visits agree exactly on the walls --
both read min=35.01, max=58.94 (wall-pressed, corroborated across visits):

```
Knight X at left wall (both visits):  35.01
Knight X at right wall (both visits): 58.94
Floor Y (lowest grounded reading across visits: 5.39, 5.40, 5.40): 5.39
Arena center X    = (35.01 + 58.94) / 2 = 46.97
Arena half-width  = (58.94 - 35.01) / 2 = 11.96
```

**Arena top / height:** visit 1's ceiling-press top was 19.09, visit 2's was
lower at 14.03; the highest across visits, 19.09, is used as the arena top
(consistent with the Hornet/Gruz/Gorb convention of taking the highest
measured top -- a tall arena covering the knight's aerial reach mid-fight),
giving:

```
Arena top (highest across visits): 19.09
Arena height = 19.09 - 5.39 = 13.70
```

**Statue:** settled menu-open readings 34.01, 34.01 (first-approach outlier
37.12 excluded). `StatueX = 34.01` -- the macro's +/-0.5 settle window
(33.51-34.51) contains all settled readings (Gruz 28.0->28.6 lesson
satisfied, so no post-hoc nudge is needed here).

**HP:** Max HP 750 Attuned (two visits, consistent), 1000 Ascended
(`GG_Mage_Knight_V`) -> mod ceiling `MaxAttunedHp = 800` (same margin style
as Gruz's 650/945 -> 700 and Gorb's 650/1000 -> 700). `TierIndex = 0`,
re-verify at this statue during smoke (registry comment convention).

**Projectile (`NeedleName` decision):** `NeedleName = null`. The only
projectile candidate in `GG_Mage_Knight` was `Hero Hurter`, logged with 3
distinct instance ids -- multiple instances, not a persistent single-instance
object, so not trackable by the mod's find-by-name mechanism. No dagger-like
single-instance candidate appeared. This is a measured absence, not a gap in
the search: the same discovery session's fixture check confirmed the logger
correctly catches persistent single-instance projectiles (Hornet's `Needle`,
instances=1), so the 3-instance reading on `Hero Hurter` reflects the boss's
actual per-shot-clone behavior. The FSM one-hot (`Shoot Antic` / `Shoot`)
still telegraphs shots -- the spec's fallback, same as Gorb's.

**Teleport discontinuities:** boss X/Y readings jump discontinuously between
reads during `Up Tele` and `Side Tele` states -- expected (the boss is
teleporting), not a reader bug. No trainer change needed.

## 10. Marmu FSM states, arena, statue, and HP

Recorded 2026-08-05 with the F4 `DiscoveryLogger`, plus per-visit analysis of
the raw ModLog, including a dedicated clean measurement visit. Scene
(Attuned): `GG_Ghost_Marmu` (Ascended: `GG_Ghost_Marmu_V`). Boss GameObject:
`Ghost Warrior Marmu`. Main FSM: **`Control`**, 5 distinct states in
first-seen order:

```
Start Pause, Antic, Chase, Unroll, Warp Out 2
```

The fifth state, `Warp Out 2`, surfaced only in the final (dedicated
measurement) visit -- the plain fight visits never showed it. A
`Broadcast Ghost Death` FSM (`Idle`) on the same object is death plumbing and
is not transcribed (same precedent as Gruz's untranscribed
`bouncer_control`).

**Arena:** walls corroborated by two independent visits reading identically
(min=51.27, max=88.73 in both the first fight visit and the dedicated
measurement visit). An intermediate fight visit read narrower, 60.53-73.02,
because the walls were never pressed during that visit -- not used:

```
Knight X at left wall (two visits agree):  51.27
Knight X at right wall (two visits agree): 88.73
Arena center X    = (51.27 + 88.73) / 2 = 70.0
Arena half-width  = (88.73 - 51.27) / 2 = 18.73
```

**Floor:** readings 10.40/10.41 across visits -- the floor is FLAT. The
spec's curved-floor caveat (grounded Y varying across the floor, requiring
the lowest settled reading plus an approximate-normalization note) did not
materialize here; `floor_y = 10.40` is exact, not an approximation.

**Arena top / height:** the first visit's ceiling-press top read 23.09; the
dedicated ceiling-press visit read lower, 22.09. Both are recorded; the
maximum across visits, 23.09, is used as the arena top (consistent with the
Hornet/Gruz/Gorb/Soul Warrior convention of taking the highest measured
top), giving:

```
Arena top (maximum across visits): 23.09
Arena height = 23.09 - 10.40 = 12.69
```

**Statue:** settled menu-open readings 91.34, 91.34 (first-approach outlier
94.52 excluded). `StatueX = 91.34` -- the macro's +/-0.5 settle window
(90.84-91.84) contains all settled readings (Gruz 28.0->28.6 lesson
satisfied, so no post-hoc nudge is needed here).

**HP:** Max HP 416 Attuned (three visits, consistent), 600 Ascended
(`GG_Ghost_Marmu_V`) -> mod ceiling `MaxAttunedHp = 450` (same margin style
as Gruz's 650/945 -> 700, Gorb's 650/1000 -> 700, and Soul Warrior's
750/1000 -> 800). `TierIndex = 0`, re-verify at this statue during smoke
(registry comment convention).

**Projectile:** `NeedleName = null`. No projectile candidates at all
appeared in `GG_Ghost_Marmu` (Attuned). A `Thorn Collider` (1 instance)
appeared only in `GG_Ghost_Marmu_V` -- an Ascended-only arena hazard,
irrelevant to the Attuned fight and not a boss projectile.

## 11. False Knight FSM states, arena, statue, and HP

Recorded 2026-08-05 with the F4 `DiscoveryLogger`, plus per-visit analysis of
the raw ModLog, including a dedicated clean measurement visit, plus the
user's identification of which fight ran at which tier. Boss GameObject:
`False Knight New`. Main FSM: **`FalseyControl`**, 54 distinct states in
first-seen order:

```
Start Fall, State 1, First Idle, Jump Antic, Rise, Fall, Idle, JA Antic,
JA Rise, JA Fall, JA Hit, JA Recoil 2, Turn R, S Attack Antic,
S Attack Recover, Run Antic, Run, JA Recoil, JA End, S Antic, S Rise,
S Fall, S Land, Stun In Air, Pause Short, Open Uuup, Opened, Hit, Recover,
Idle Pause, Rage Jump Antic, Rise 2, Fall 2, State 2, R Attack Antic,
Rage, Particle Pause, Anim End, Stun Land, Rage End, Death Open, Opened 2,
Hit 2, Death Anim Start, Steam, Ready, Blow, Death Head Land, Cough,
S Attack, Slam, Stun Fail, Turn L, JA Slam
```

By far the largest state list of any registered boss so far -- transcribed
carefully against the parser output, since a single dropped or reordered
state would silently corrupt the observation one-hot.

Two secondary FSMs are present but not transcribed into the state list: a
`Check Health` FSM (`Check`) on `False Knight New` itself, and a separate
`Head` GameObject (hp=40, FSM `Health Check`: `Check 1`) present in every
fight at both tiers -- the stagger-phase target that gets exposed when the
boss's armor breaks. Win detection is the `Die` hook on `False Knight New`;
the smoke run confirms it fires through his death sequence (same convention
as the untranscribed secondaries on Gruz, Gorb, Soul Warrior, and Marmu).

**Scene / tier (important -- differs from every other registered boss):**
Scene is `GG_False_Knight` for BOTH Attuned and Ascended -- the user's third
fight was Ascended and still logged `scene=GG_False_Knight`; no `_V` scene
exists anywhere in the log. Every other boss registered so far gets an
Ascended `_V` scene that backstop A (scene match) can use to reject a
wrong-tier fight; False Knight has none. Consequence: for this boss, the HP
ceiling (backstop B, `MaxAttunedHp`) is the ONLY wrong-tier guard -- scene
matching alone cannot tell Attuned and Ascended apart. This is recorded here
and repeated as a comment on the registry's `MaxAttunedHp` field.

**Arena:** walls corroborated by two independent visits reading identically
(min=11.19, max=45.70 in both the first fight visit and the dedicated
measurement visit). Two intermediate fight visits read narrower --
12.49-30.13 and 23.78-45.37 -- because the walls were never pressed during
those visits; not used:

```
Knight X at left wall (two visits agree):  11.19
Knight X at right wall (two visits agree): 45.70
Arena center X    = (11.19 + 45.70) / 2 = 28.45
Arena half-width  = (45.70 - 11.19) / 2 = 17.26
```

**Floor:** 27.40 identical in every visit -- the floor is FLAT.

**Arena top / height:** the first visit's ceiling-press top read 42.81; the
dedicated ceiling-press visit read lower, 40.71. Both are recorded; the
maximum across visits, 42.81, is used as the arena top (consistent with the
Hornet/Gruz/Gorb/Soul Warrior/Marmu convention of taking the highest
measured top), giving:

```
Arena top (maximum across visits): 42.81
Arena height = 42.81 - 27.40 = 15.41
```

**Statue:** settled menu-open readings 52.07, 52.07, 52.07 (no outliers).
`StatueX = 52.07` -- the macro's +/-0.5 settle window (51.57-52.57) contains
all settled readings (Gruz 28.0->28.6 lesson satisfied, so no post-hoc nudge
is needed here).

**HP:** Max HP 260 Attuned (three sightings across both sessions,
consistent), 560 Ascended (user-identified third fight, same scene as
Attuned -- see the scene/tier note above) -> mod ceiling
`MaxAttunedHp = 300` (above 260, below 560). Because this boss has no
Ascended `_V` scene, this ceiling is the load-bearing wrong-tier guard, not
just a backstop alongside scene matching. `TierIndex = 0`, re-verify at this
statue during smoke (registry comment convention).

**Projectile:** `NeedleName = null`. All `GG_False_Knight` projectile
candidates are per-shot clones: `Falling Barrel(Clone)` (24 instances),
`Shockwave Spurt(Clone)` (52 instances), `Shockwave Spurt L(Clone)` (80
instances). None is a stable single-instance object, so none is trackable by
the mod's find-by-name mechanism (same pattern as Gorb's `Shot Slug
Spear(Clone)` and Soul Warrior's `Hero Hurter`). The falling debris in
particular reads as projectile-shaped but the high instance count marks it
as per-drop clones, not a persistent object worth tracking.

---

## 12. Workshop statue levels and stand Y

`GG_Workshop` (the Godhome hub) is not a single flat floor -- it has two
levels. A smoke-gorb run surfaced this the hard way: the reset macro walked
to the ground-floor Gorb `StatueX` (x=126) and stood there pressing Up, but
the challenge menu never opened -- every 22.5s reset budget expired with the
Knight stuck at ground level while the actual Gorb statue sits one level up
(2026-08-05). Investigating confirmed the workshop has a ground floor
(Hornet, Gruz Mother, and False Knight's statues) and an upper walkway
(Gorb, Marmu, and Soul Warrior's statues), and that walking alone cannot
cross between them -- a fix needs a teleport, not a longer walk.

A dedicated measurement session (2026-08-05, after the knightY logger
change) read the statue-stand positions with Y:

```
Gorb:         knightX=126.21  knightY=36.41
Marmu:        knightX=91.35   knightY=36.41
Soul Warrior: knightX=37.19   knightY=36.41
False Knight: knightX=55.19   knightY=6.41   (ground floor confirmed)
```

The upper walkway is flat: all three upper stands read y=36.41. False
Knight's reading, y=6.41, matches the ground floor's measured 6.40, so he
(along with Hornet and Gruz Mother, also ground-floor) keeps the `StatueY`
NaN default and the proven walk-only path -- no upper-level teleport is
needed or set for those three.

**Teleport mechanism:** `BossSpec.StatueY` defaults to `float.NaN`. When a
boss's `StatueY` is set (not NaN), the reset macro teleports the Knight to
`(StatueX, StatueY)` once per reset attempt before approaching the statue,
instead of relying on the walk-only navigation used by the ground-floor
bosses. `gorb`, `marmu`, and `soul_warrior` now carry `StatueY = 36.41f`,
using this mechanism to reach the upper walkway.

**X-spread observation:** this same measurement session re-read `StatueX`
for two bosses already registered and found the interact region spans
several units, not a single point -- Soul Warrior's menu opened at both
34.01 and 37.19, False Knight's at both 52.07 and 55.19. The registered
`StatueX` values (34.01 for Soul Warrior, 52.07 for False Knight) remain
the settled multi-reading choices from their original discovery sessions
(sections 9 and 11) and are unchanged by this session's readings.
