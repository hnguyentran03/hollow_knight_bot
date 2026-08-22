# Hollow Knight RL Bot

A reinforcement-learning bot that learns to fight **Hall of Gods bosses (Attuned tier)** in Hollow Knight, trained with PPO against the real game running in lockstep. Six bosses are registered: `hornet1`, `gruz_mother`, `marmu`, `false_knight`, and `soul_warrior` all trained to winning policies, plus `gorb` (43% win rate).

Two halves:

- **`mod/`** — a C# game mod (HKRLBot) running inside Hollow Knight. It exposes a TCP bridge that accepts button actions, holds them for one 67&nbsp;ms tick, samples the game state, and sends it back. It drives all episode resets itself — from a fresh boot it can walk the title menu, reach the chosen boss's statue, and start the fight with no human input. F1 shows an in-game HUD of what the mod sees; a Mods-menu selector plus F9 let a trained, exported bot take over your game for one fight.
- **`trainer/`** — a Python package (`hkrl`) exposing the mod as a Gymnasium environment, plus scripts for training (RecurrentPPO), a web dashboard, replaying checkpoints, and smoke-testing.

## Architecture

```
┌────────────────────────┐   {"type":"action","buttons":{...}}   ┌───────────────────────────┐
│  trainer (Python)      │ ────────────────────────────────────► │  Hollow Knight + HKRLBot  │
│  PPO / random agent    │                                       │  holds buttons 67 ms,     │
│                        │ ◄──────────────────────────────────── │  samples state            │
└────────────────────────┘   {"type":"state","obs":{...},...}    └───────────────────────────┘
```

One decision every 67&nbsp;ms (15&nbsp;Hz). The agent picks one of **21 discrete moves** (walk, jump, slash, dash, pogo, spells, Focus, directional combinations); buttons stay held across repeated steps, so jump height, healing, and nail-art charging are emergent. Observations are 18 normalized scalars plus a one-hot of the boss's FSM state, sized per boss — so **checkpoints are boss-specific**; the policy is a recurrent LSTM. Reward: boss damage +0.03/HP, hits −1/mask, win +10 (+1/mask remaining), death −5, small per-step time penalty.

Training is fault-tolerant end to end: the trainer launches and owns the game, reconnects through the mod's normal reset-budget drops, relaunches wedged or crashed instances (re-seeding their save from the master on macOS, so a bad save self-heals), and saves a checkpoint ("generation") every 15k steps — a crash never loses more than ~17 minutes.

## Prerequisites

- macOS or Windows with Hollow Knight installed via Steam at the default location:
  - macOS: `~/Library/Application Support/Steam/steamapps/common/Hollow Knight/`
  - Windows: `C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight\`
- The [Hollow Knight Modding API](https://github.com/hk-modding/api) installed
- .NET SDK (for `dotnet build`)
- Python 3.11+

Shell commands below are for macOS; on Windows use PowerShell with `.venv\Scripts\python` in place of `./.venv/bin/python`, and omit the `caffeinate` prefix. For a non-default Steam library, pass `-HKManaged <path>` to run.ps1 or edit `HK_APP` at the top of run.sh.

**One-time game prep** (the only step no script can do for you): park the training save in Godhome with the bosses you plan to train unlocked in the Hall of Gods, ideally resting at the Hall of Gods bench. Every start automatically snapshots the save directory to `~/hkrl/save-backups/` (newest 10 kept).

## Starting via the dashboard

One command checks dependencies (reporting everything missing at once), creates the Python env, builds/installs/re-signs the mod when its source changed, and opens the dashboard:

```bash
bash run.sh                                          # macOS
powershell -ExecutionPolicy Bypass -File run.ps1     # Windows
```

From the dashboard's **"Summon a run" panel** you can start, resume, and stop training without a terminal. It spawns `train.py --auto` detached (so the run survives a dashboard exit, and a restarted dashboard picks it back up); `--auto` lets the boot macro drive the game in — a few `reset ... reconnecting` retries in the log tail are normal. The launch form has a boss dropdown and advanced PPO fields (blank Target KL keeps the 0.03 default), plus a Headless checkbox. One launched run at a time; Stop is the same graceful path as Ctrl-C. Runs started from a terminal are invisible to the panel — stop those in their own terminal. Deleting a run moves it to `~/hkrl/trash/`, never erases it.

The dashboard (`http://127.0.0.1:9700`) also monitors every run: live/stopped status, progress and ETA, steps/hour, total wins, learning curves, per-episode reward. It is read-only and safe beside a live run. Lower-level health checks:

```bash
tail -f ~/hkrl/runs/my-run/generations.jsonl                  # per-generation stats
./.venv/bin/tensorboard --logdir ~/hkrl/runs/my-run/tb
```

## Starting via the CLI

Unless running `--headless`, keep the **game window visible** for the whole run (macOS suspends occluded windows — App Nap — and every occurrence costs a relaunch; on Windows don't minimize), and set display-off/screen-saver to Never or use `caffeinate` as below.

```bash
cd trainer
caffeinate -dims ./.venv/bin/python scripts/train.py --timesteps 500000 --run-id my-run
```

`train.py` launches the game itself — don't start one manually. When prompted, wait for the game to reach the Hall of Gods (the boot macro can drive it there) and press Enter. 500k steps is roughly an overnight run (~54k steps/hour at 15&nbsp;Hz).

Useful flags: `--boss` (default `hornet1`; ids from `hkrl/bosses.py`), `--instances N`, `--gen-every` (checkpoint interval, default 15000), `--n-steps` / `--batch-size` / `--n-epochs` (fresh runs only), `--target-kl` (early-stops an update when approx KL exceeds ~1.5× this; fresh runs default 0.03, 0 disables; resumes inherit the run's recorded value, and typing the flag on a resume overrides it), `--seed`, `--root` (default `~/hkrl`), `--headless` (no game window: `-batchmode -nographics`, the mod caps the frame loop at 60 fps; a resume keeps the last session's value, and `--headless`/`--no-headless` override it).

**Multiple instances:** `--instances N` runs N game clients as slots of one vectorized PPO — roughly N× the samples per hour, verified live at N=2 headed and N=4 headless (~184k steps/hour, double the N=2 rate, learning unaffected). Each headed instance's window must stay visible (tile them, don't stack them), which caps headed runs at 2–3 per machine; `--headless` removes that constraint entirely, which is what makes N=4 practical. Episode resets run asynchronously by default at N ≥ 2 (`--no-async-resets` restores lockstep). On macOS each instance gets its own cloned app and save sandbox automatically, re-seeded from the master save every start; **Windows has no save isolation yet** — multi-instance there shares one save slot at your own risk.

**Pause and resume:** Ctrl-C once finishes the current episode (≤ ~3 min), saves a final generation, and shuts the game down (twice forces an abort, still saving). Resume continues the same run — weights, normalization stats, generation numbering, and the recorded config (`--instances`, `--target-kl`, `--headless`, ports, boss) are all inherited from the last session; omitting `--timesteps` finishes to the run's recorded target, `--timesteps N` adds N more:

```bash
caffeinate -dims ./.venv/bin/python scripts/train.py --resume ~/hkrl/runs/my-run
```

## Watching and playing bots

None of these may run against a live trainer's port — the mod keeps only its newest client.

**Smoke test** — verify the pipeline with a random policy (typically deals ~30–45% boss damage):

```bash
./.venv/bin/python scripts/launch_instances.py       # launches a game manually
./.venv/bin/python scripts/random_agent.py --episodes 2
```

**Replay a checkpoint** — launch a game as above, then:

```bash
./.venv/bin/python scripts/replay.py --run-dir ~/hkrl/runs/my-run --episodes 3          # latest gen
./.venv/bin/python scripts/replay.py --run-dir ~/hkrl/runs/my-run --gen 1 --episodes 3
```

**Play a bot in-game with F9** — drops a trained bot into *your* game for exactly one fight. Export a generation (the Export button on any generation row in the dashboard, or `scripts/export_gen.py --run-dir ~/hkrl/runs/my-run`), start the playback daemon (`./.venv/bin/python scripts/play.py`) and the game in either order, pick the bot under **Options → Mods → HKRLBotMod**, stand in the Hall of Gods, and press **F9**. The daemon walks the Knight to the right statue, fights one episode, and hands control back; the F1 HUD's BOT chip shows its state.

**Adding a boss** is a measurement job, not a coding one: press F4 in a normal game session and fight the new boss a few times — the mod's discovery logger records boss candidates, FSM transitions, arena extremes, statue positions, and projectile candidates to ModLog. `scripts/parse_discovery.py <ModLog path>` reduces that to registry-ready values, transcribed into `hkrl/bosses.py` and `mod/BossRegistry.cs` following `mod/DISCOVERED.md`.

## Notes

- After a manual `mod/build.sh` on macOS, re-sign or the game exits with code 138: `codesign --force --deep --sign - "$HOME/Library/Application Support/Steam/steamapps/common/Hollow Knight/hollow_knight.app"` (`run.sh` does this automatically).
- If `train.py` fails fast naming an occupied bridge port (9020+), something else holds it — often an old dashboard or a leftover game instance.
