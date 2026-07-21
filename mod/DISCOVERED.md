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

**Fix:** `StateReader.cs` adds `IsChallengeMenuOpen()`
(`FindObjectOfType<BossChallengeUI>() != null`, exception-safe), letting
`EpisodeManager`'s gate split the two cases the old code couldn't tell
apart:
- menu not open: proceed unchanged, no log (this fires on every normal
  pre-open window and previously spammed a "gate inactive" line for no
  reason).
- menu open, tier `0`: allow the confirm (silent, healthy path).
- menu open, tier `>0`: correct to Attuned, veto the confirm (unchanged).
- menu open, tier `-1`: call `SelectAttunedChallengeTier()` and veto the
  confirm, up to 2 attempts per statue visit (logged per attempt). If still
  `-1` after 2 attempts, fall back to the blind confirm with one loud log
  line -- the gate never withholds the confirm indefinitely, since starving
  the macro turns a readability problem into a dead run (budget expiry ->
  `InstanceDown`), and the blind confirm is the pre-gate behavior already
  verified to land Attuned on a fresh process.

`SetSelectedGameObject` is focus-independent (it writes engine state
directly rather than relying on input-driven UI navigation), which is why
the mod can select Attuned itself even though the game never does so on its
own in this environment.
