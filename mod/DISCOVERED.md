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

**Complete.** Recorded from live play on 2026-07-18. Derived values verified
arithmetically against the measured walls. Tasks 6 and 7 may now consume these
numbers.
