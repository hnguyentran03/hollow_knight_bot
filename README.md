# Hollow Knight RL Bot

A reinforcement-learning bot that learns to fight **Hall of Gods bosses (Attuned tier)** in Hollow Knight — six registered so far, from Hornet Protector to False Knight — trained with PPO against the real game running in lockstep.

The project has two halves:

- **`mod/`** — a C# game mod (HKRLBot) that runs inside Hollow Knight. It exposes a TCP bridge that accepts button actions, holds them for one 67&nbsp;ms tick, samples the game state (Knight + boss positions, velocities, HP, SOUL, the boss's FSM state), and sends it back. It also drives all episode resets itself: from a fresh boot it can walk through the title menu, stand up from the Hall of Gods bench, get to the chosen boss's statue, and start the fight — no human input needed. An in-game HUD styled after the game's own UI (F1) shows what the mod sees: HP/SOUL bars, boss name and HP, FSM state, projectile tracking.
- **`trainer/`** — a Python package (`hkrl`) exposing the mod as a Gymnasium environment, plus scripts for training (RecurrentPPO from sb3-contrib), replaying checkpoints, and smoke-testing with a random agent.

## How it works

```
┌────────────────────────┐   {"type":"action","buttons":{...}}   ┌───────────────────────────┐
│  trainer (Python)      │ ────────────────────────────────────► │  Hollow Knight + HKRLBot  │
│  PPO / random agent    │                                       │  holds buttons 67 ms,     │
│                        │ ◄──────────────────────────────────── │  samples state            │
└────────────────────────┘   {"type":"state","obs":{...},...}    └───────────────────────────┘
```

One decision every 67&nbsp;ms (15&nbsp;Hz). The agent picks one of **21 discrete moves** (walk, jump, slash, dash, pogo, spells via Quick Cast, Focus — including directional combinations); buttons stay held across consecutive steps that repeat them, so jump height, healing, and nail-art charging are emergent. Observations are 18 normalized scalars plus a one-hot of the boss's FSM state, sized per boss (56 floats against Hornet Protector, 36 against Gruz Mother, ranging down to 24 for Marmu and up to 73 for False Knight); the policy is a recurrent LSTM, so it carries its own memory instead of frame stacking. Reward is dominated by boss damage dealt (+0.03/HP), hits taken (−1/mask), win (+10), death (−5), and a small per-step time penalty.

Training is fault-tolerant end to end: the trainer launches and owns the game process, reconnects through the mod's normal reset-budget drops, and if the game wedges or crashes it relaunches it and keeps training — every relaunch re-seeds that instance's save from the master (macOS), so even a corrupted or wrong save self-heals; the trainer detects those boots by their reset-abort pattern and fails fast into that recovery instead of burning its retry ceiling. Checkpoints ("generations") are saved every 15k steps, so a crash or Ctrl-C never loses more than ~17 minutes.

## Repository layout

```
mod/                    C# mod source
  BridgeServer.cs         TCP server, one JSON message per line
  EpisodeManager.cs       lockstep action/state cycle + reset macro (boot → bench → statue → fight)
  StateReader.cs          reads Knight/boss state from the live game
  VirtualInput.cs         virtual controller the agent's buttons drive
  DISCOVERED.md           measured game facts (FSM state names, arena bounds, ...)
  build.sh                builds and installs the mod into the game (macOS)
  build.ps1               the same, for Windows
trainer/
  hkrl/                   the Python package
    env.py                  Gymnasium env: actions, observations, reward
    protocol.py             line-JSON socket protocol
    game.py                 launches/kills the game process
    supervisor.py           VecEnv wrapper that recovers from crashes/wedges
    generations.py          checkpointing + manifest + resume
    rundata.py              read-only parsing/aggregation of run directories
    dashboard.py            HTTP server for the web dashboard (+ dashboard.html)
    fake_game.py            scripted in-process game for tests
  scripts/
    train.py                the training entry point
    dashboard.py            serve the web dashboard (default port 9700)
    random_agent.py         random-policy smoke test (game must already be running)
    replay.py               watch a saved generation play
    launch_instances.py     manually launch a game instance (for random_agent/replay)
  tests/                  pytest suite (real sockets against fake_game, no mocks)
```

## Prerequisites

- macOS or Windows with Hollow Knight installed via Steam at the default location:
  - macOS: `~/Library/Application Support/Steam/steamapps/common/Hollow Knight/`
  - Windows: `C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight\`
- The [Hollow Knight Modding API](https://github.com/hk-modding/api) installed (the mod links against `MMHOOK_Assembly-CSharp.dll`)
- .NET SDK (for `dotnet build`)
- Python 3.11+

Shell commands below are written for macOS. On Windows they work the same
from PowerShell with the venv paths swapped: `.venv\Scripts\python` instead
of `./.venv/bin/python` (and likewise for `pip` and `tensorboard`). The
trainer's `--app`/`--root` flags accept Windows paths directly.

## Quick start

One command sets everything up and opens the dashboard, first run and every
run after — it checks dependencies (and says exactly what's missing), creates
the Python env, builds/installs/re-signs the mod when its source changed, and
starts the dashboard, from whose "Summon a run" panel you can start training:

```bash
bash run.sh                                          # macOS
powershell -ExecutionPolicy Bypass -File run.ps1     # Windows
```

Extra arguments are passed to the dashboard (e.g. `bash run.sh --port 9701`).
For a non-default Steam library, pass `-HKManaged <path>` to run.ps1 or edit
`HK_APP` at the top of run.sh.

**One-time game prep** (the only step no script can do for you): the training
save file should be parked in Godhome with the bosses you plan to train
unlocked in the Hall of Gods, ideally resting at the Hall of Gods bench. The
mod's boot macro handles everything from the title screen onward.

## Smoke test: the random agent

Before a long run, verify the whole pipeline with a random policy. Launch the game first, then:

```bash
cd trainer
./.venv/bin/python scripts/launch_instances.py   # or launch the game yourself
./.venv/bin/python scripts/random_agent.py --episodes 2
```

You should see the fight start on its own and one summary line per episode (steps, reward, boss damage %). Random play typically deals ~30–45% boss damage before dying.

## Training

### Before you start

- Keep the **game window visible** for the entire run (on Windows: not minimized). macOS suspends fully occluded windows (App Nap) even with their port open; every occurrence costs a full relaunch-and-recovery.
- Set screen saver and display-off to **Never** (macOS can instead rely on `caffeinate` below; on Windows use Settings → System → Power, or `powercfg /change standby-timeout-ac 0`).
- Have a few hundred MB free under `~/hkrl` for checkpoints and logs.

### Start a run

```bash
cd trainer
caffeinate -dims ./.venv/bin/python scripts/train.py --timesteps 500000 --run-id my-run
```

(On Windows, run `.venv\Scripts\python scripts\train.py ...` without the `caffeinate` prefix — sleep suppression comes from the power settings above.)

### Multiple instances

`--instances N` runs N game instances in parallel (bridge ports `--port`
through `--port + N - 1`), each a slot of the same vectorized PPO — roughly
N× the samples per hour. The supervisor recovers each slot independently,
exactly as it does at N=1. Verified live at N=2: an async two-instance run
reached a higher reward than the single-instance baseline in less
wall-clock time (2026-07-26).

```bash
caffeinate -dims ./.venv/bin/python scripts/train.py --instances 2 --timesteps 500000 --run-id my-run
```

What to know before trying it:

- **Every instance is a full game client.** 2–3 on one machine is realistic;
  each window must stay visible (see App Nap above), so tile them, don't
  stack them.
- **`--n-steps` is per instance and its default divides by N** (so the
  total batch, and with it the PPO update's wall-clock time, stays roughly
  constant), then inflates ~23% when async resets are on so each update
  still sees ~2048 *real* samples after the placeholder rows are masked
  out. The keepalive pinger keeps connections alive through the update,
  but the games keep running in real time while it computes — every Knight
  stands in a live fight for the duration — so if you override
  `--n-steps`, keep the total real batch around 2048.
- **Episode resets run asynchronously by default at N ≥ 2** (the mod's
  reset macro can run its full 22.5 s budget; synchronously it would
  freeze every sibling mid-fight for that long). The resetting slot feeds
  the learner placeholder frames — masked out of the gradient, the
  monitoring, and the normalization statistics — while the others keep
  taking real steps. `--no-async-resets` restores the old lockstep
  behavior; expect episode throughput somewhat below N× with it.
- **Each instance gets its own save/log sandbox (macOS).** Instances
  sharing one save directory autosave the same slot concurrently, which
  corrupted the master save in live testing (both games saved in the same
  second; the next boot read the slot as empty and started a new game — the
  game's own `user1.dat.bakNNN` rotation recovered it). The trainer clones
  the whole `.app` per port — every instance, N=1 included — under
  `<root>/instances/port-<port>/` (APFS copy-on-write: instant, near-zero
  disk) with a per-port bundle identifier, which moves that instance's
  Unity save directory and `ModLog.txt` wholesale; each clone's save dir is
  seeded from the master save at every start and re-seeded on every
  relaunch. Training never opens the
  master save or app, in-run save churn is disposable, and a mod rebuild on
  the master propagates to the clones on the next start. Keep the master
  save parked at the Hall of Gods bench; it is copied, not played. (`HOME`
  redirection was tried first and disproven live — Unity ignores it.)
  **Windows has no save isolation yet** — multi-instance there shares one
  slot at your own risk, and the trainer warns accordingly.
- **Two direct-exec Steam instances at once is verified live on macOS**
  (2026-07-20: both processes survive the DRM window and both bridges serve
  the protocol simultaneously). Windows multi-instance is unverified; if an
  instance quits ~15 s after boot there, that's the DRM check — report it
  before working around it.
- Replay and the random agent are single-instance tools: point them at one
  port (`--port`) of a manually launched game
  (`launch_instances.py --instances N` launches a tiled set for that).

`train.py` launches the game itself — don't start one manually. When prompted, wait for the game to reach the Hall of Gods (the boot macro can drive it there; a few `reset ... reconnecting` retries on stderr are normal) and press Enter. 500k steps is roughly an overnight run (~54k steps/hour at 15&nbsp;Hz).

Useful flags: `--gen-every` (checkpoint interval, default 15000), `--n-steps` / `--batch-size` / `--n-epochs` (PPO rollout/update shape), `--target-kl` (early-stop an update's remaining epochs once approx KL exceeds ~1.5× this; off by default — try 0.05 if a long run peaks then slides), `--boss` (which boss to train against, default `hornet1`; picks from the registry in `hkrl/bosses.py` — currently `hornet1`, `gruz_mother`, `marmu`, `false_knight`, and `soul_warrior`, all trained to a winning policy, plus `gorb`, trained to a 43% win rate), `--seed`, `--root` (default `~/hkrl`).

A boss sets the observation space, so checkpoints are boss-specific — a resume always keeps the run's recorded boss (read from `config.jsonl`, even for runs from before `--boss` existed, which default to `hornet1`) and refuses a conflicting `--boss` flag up front, rather than failing on a shape mismatch deep inside model load. Growing a boss's registered FSM state list changes its observation size the same way, so runs from before such an addition can't resume either (hornet1's list grew in 2026-08 when the trainer's unseen-state warning surfaced ten states missing from the original measurement).

Adding a boss is a measurement job, not a coding one: press F4 in a normal (human-played) game session and the mod's discovery logger records boss candidates with HP, every FSM's state transitions, the knight's arena extremes, and the statue-stand X to ModLog while you fight the new boss a few times; `scripts/parse_discovery.py <ModLog path>` then reduces that to registry-ready values, which get transcribed into `hkrl/bosses.py` (trainer side) and `mod/BossRegistry.cs` (mod side) following `mod/DISCOVERED.md`. The logger also records enemy-projectile candidates (hero-damaging objects owned by no enemy, instance counts distinguishing a persistent trackable object from per-shot clones) so `ProjectileName` comes from measurement. Statue lines record X and Y: the Hall of Gods workshop is two-level, and for a boss whose statue sits on the upper walkway the reset macro teleports the knight to the measured stand (`StatueX`/`StatueY`) instead of walking blind on the wrong floor. Gruz Mother went through exactly this pipeline and trained to a sustained 100% win rate inside a single 500k-step run (~90% by 120k steps, a solid 1.0 from ~250k on); `gorb`, `marmu`, `false_knight`, and `soul_warrior` followed the same pipeline and each got a 500k-step run with no unseen states. Three trained to winning policies — False Knight 100% win rate (his win-dominated reward converged even though armor hits never move his tracked HP pool), Marmu ~96%, Soul Warrior 90% — while Gorb reached 43% with ~91% mean boss damage, learning strongly without converging in the budget.

### Monitor progress

The web dashboard shows every run under `~/hkrl/runs/`: live/stopped status,
timestep progress and ETA, steps/hour, total wins, the learning curves (boss
damage, win rate, reward, episode length per generation), and a per-episode reward chart
that updates between checkpoints. It is read-only — it never touches the game
port — so it is safe to leave up beside a live run:

```bash
./.venv/bin/python scripts/dashboard.py --open   # http://127.0.0.1:9700
```

Each run lives in `~/hkrl/runs/<run-id>/`. The per-generation manifest is the health record:

```bash
tail -f ~/hkrl/runs/my-run/generations.jsonl
```

Healthy: `episodes` > 0 each generation, `mean_boss_damage` trending up, `recoveries` rare. Warning signs: `episodes: 0`, `recoveries` climbing every generation (App Nap — check window occlusion), or episode length pinned at 2700 with zero damage. TensorBoard logs are in the run's `tb/` directory:

```bash
./.venv/bin/tensorboard --logdir ~/hkrl/runs/my-run/tb
```

### Launching runs from the dashboard

The dashboard's "Summon a run" panel can start, resume, and stop training
without a terminal: it spawns `train.py --auto` detached (wrapped in
`caffeinate -dims` on macOS), so the run keeps going if the dashboard
exits, and a restarted dashboard picks the live run back up. `--auto`
skips the Hall-of-Gods prompt and lets the boot macro drive the game in —
expect a few `reset ... reconnecting` retries in the panel's log tail.

One launched run at a time (the bridge ports allow no more). Stop is the
same graceful path as Ctrl-C: finish the episode, save a final
generation, shut the games down. Console output and pidfiles live under
`~/hkrl/launcher/`; run directories are still written only by `train.py`.
Runs started from a terminal are not tracked by the panel — stop those in
their own terminal.

### Pause and resume

**Pause:** press Ctrl-C once. The run finishes the current episode (≤ ~3 minutes), saves a final generation, and shuts the game down. A second Ctrl-C forces an immediate abort (it still attempts a save).

**Resume:**

```bash
caffeinate -dims ./.venv/bin/python scripts/train.py --resume ~/hkrl/runs/my-run
```

This relaunches the game, reloads the latest generation's weights **and** its observation-normalization statistics, and continues the same run — same directory, continued generation numbering and TensorBoard curves. A resume inherits the run's recorded settings (`--instances`, `--gen-every`, `--target-kl`, the async-reset flags, `--port`) from its `config.jsonl`, with any explicitly passed flag still winning. Omitting `--timesteps` finishes to the run's recorded target (a run already at its target refuses to start); passing `--timesteps N` stays additive ("collect N more"). If a run dies overnight (recovery exhausted), it prints this exact `--resume` command before exiting. PPO hyperparameters come from the checkpoint on resume, so the `--n-steps`/`--batch-size`/`--n-epochs`/`--seed` flags shape fresh runs only and are refused on resume. The one exception is `--target-kl`, which when passed overrides the checkpoint's value too — it exists to change update dynamics on a run already in progress.

### Watch a checkpoint play

After training stops (never against a port a live trainer is using), launch a game instance and replay any generation:

```bash
./.venv/bin/python scripts/launch_instances.py
./.venv/bin/python scripts/replay.py --run-dir ~/hkrl/runs/my-run --episodes 3          # latest gen
./.venv/bin/python scripts/replay.py --run-dir ~/hkrl/runs/my-run --gen 1 --episodes 3  # a specific gen
```

Comparing an early generation against the latest is the quickest way to *see* what it has learned.

## Things that bite

- **Forgot to re-sign after a mod build (macOS)** → game exits immediately with code 138. `run.sh` re-signs automatically after every build; if you ran `mod/build.sh` by hand, follow it with `codesign --force --deep --sign - "$HOME/Library/Application Support/Steam/steamapps/common/Hollow Knight/hollow_knight.app"`.
- **Occluded game window (macOS) / minimized window (Windows)** → the OS suspends or deprioritizes the game; the trainer sees a wedge and burns a recovery. Keep it visible, even if small.
- **Old checkpoints after changing the action/observation space** → not resumable or replayable; the policy network's shape changed. Start a fresh run.
- **The mod drops connections that go silent for 10 s** → the trainer's keepalive pinger (`hkrl/protocol.py`) keeps every connection chatty through lockstep gaps (another instance's reset, a PPO update), so this no longer severs connections. The game still runs in real time while the trainer thinks, though — a slow update leaves the Knight standing in a live fight — so still lower `--n-epochs` if updates get slow; don't raise the mod's `ReadTimeout`.
- **Worried about your save?** Every `train.py` / `launch_instances.py` start automatically snapshots the whole save directory to `~/hkrl/save-backups/<timestamp>/` (newest 10 kept). To restore, quit the game and copy a snapshot's contents back over the save directory. The game's own `.bakNNN` rotation is not enough on its own — it lives inside the directory it would need to protect.
- **Deleting a run from the dashboard** moves its directory to `~/hkrl/trash/<id>-<timestamp>/` rather than erasing it — a mis-click never costs you checkpoints. The list forgets it immediately; empty `trash/` yourself when you're sure. A run that is live, or that any process has touched in the last five minutes (a terminal-started session the dashboard can't see), is refused.
- **Something else holds a bridge port** → the mod's listener silently fails and the game runs bridgeless, while port probes greet the squatter. `train.py` and `launch_instances.py` both fail fast on this before launching; if they name a port you recognize, it's probably an old dashboard (its default was 9021 — the second instance's bridge port — before it moved to 9700).
