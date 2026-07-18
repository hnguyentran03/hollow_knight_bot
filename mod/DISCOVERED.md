# DISCOVERED.md — values recorded from live play with the echo overlay

This file records ground-truth numbers read off the on-screen overlay (Task 2,
`OverlayUI.cs`, toggled with **F1**) while actually playing the game. It is a
template. **Every value below is currently unfilled and marked `TBD`.**

Do not fill in a value unless you personally read it off the overlay while
playing. Do not estimate, guess, or infer a "plausible" number — later tasks
(the Python trainer's `HORNET_STATES` list in Task 7, the arena bounds used
for normalizing observations, and `StatueX` in Task 6) will consume these
values verbatim and a wrong-but-plausible number here will silently corrupt
the bot's observations with no visible error.

Every unfilled slot is marked with `<!-- TO FILL: ... -->` and/or the literal
text `TBD`. Do not remove these markers until you have replaced them with a
real observed value.

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
<!-- TO FILL: list every distinct `state=` value observed on the Boss line,
     one per line, exactly as printed by the overlay. -->
TBD — fill during verification pass
```

## 2. Arena bounds (Hall of Gods, Hornet 1 arena)

Walk the Knight to the left wall of the arena and note the `x=` value on the
`Knight:` line. Do the same at the right wall. Also note the `y=` value while
standing on the arena floor (not jumping/falling).

```
Knight X at left wall:   TBD  <!-- TO FILL: overlay Knight x= value -->
Knight X at right wall:  TBD  <!-- TO FILL: overlay Knight x= value -->
Floor Y:                 TBD  <!-- TO FILL: overlay Knight y= value while standing on the floor -->
```

Derived values (compute only after both X values above are filled in):

```
Arena center X     = (left + right) / 2   = TBD
Arena half-width X = (right - left) / 2   = TBD
```

## 3. Statue-stand X in `GG_Workshop`

In `GG_Workshop` (the Godhome hub room), stand at the Hornet statue and note
the Knight's `x=` value from the overlay.

```
Knight X at Hornet statue in GG_Workshop: TBD  <!-- TO FILL: overlay Knight x= value; feeds StatueX in Task 6 -->
```

---

## Status

- [ ] Hornet FSM state list recorded
- [ ] Arena X range recorded (both walls)
- [ ] Floor Y recorded
- [ ] Arena center / half-width computed
- [ ] Statue X recorded in GG_Workshop

**No values have been filled in yet. This file is a template only, produced
during Task 2 implementation without in-game access. Fill in the sections
above during a verification play session before any later task consumes these
numbers.**
