# Hollow Knight RL Bot

A reinforcement-learning bot that learns to fight **Hornet (Hall of Gods, Attuned)** in Hollow Knight, trained with PPO against the real game running in lockstep.

The project has two halves:

- **`mod/`** — a C# game mod (HKRLBot) that runs inside Hollow Knight. It exposes a TCP bridge that accepts button actions, holds them for one 67&nbsp;ms tick, samples the game state (Knight + Hornet positions, velocities, HP, SOUL, Hornet's FSM state), and sends it back. It also drives all episode resets itself: from a fresh boot it can walk through the title menu, stand up from the Hall of Gods bench, run to the Hornet statue, and start the fight — no human input needed.
- **`trainer/`** — a Python package (`hkrl`) exposing the mod as a Gymnasium environment, plus scripts for training (Stable-Baselines3 PPO), replaying checkpoints, and smoke-testing with a random agent.

## How it works

```
┌────────────────────────┐   {"type":"action","buttons":{...}}   ┌───────────────────────────┐
│  trainer (Python)      │ ────────────────────────────────────► │  Hollow Knight + HKRLBot  │
│  PPO / random agent    │                                       │  holds buttons 67 ms,     │
│                        │ ◄──────────────────────────────────── │  samples state            │
└────────────────────────┘   {"type":"state","obs":{...},...}    └───────────────────────────┘
```

One decision every 67&nbsp;ms (15&nbsp;Hz). The agent picks one of **21 discrete moves** (walk, jump, slash, dash, pogo, spells via Quick Cast, Focus — including directional combinations); buttons stay held across consecutive steps that repeat them, so jump height, healing, and nail-art charging are emergent. Observations are 46 floats (18 normalized scalars + a 28-way one-hot of Hornet's FSM state), frame-stacked ×4. Reward is dominated by boss damage dealt (+0.03/HP), hits taken (−1/mask), win (+10), death (−5), and a small per-step time penalty.

Training is fault-tolerant end to end: the trainer launches and owns the game process, reconnects through the mod's normal reset-budget drops, and if the game wedges or crashes it relaunches it and keeps training. Checkpoints ("generations") are saved every 15k steps, so a crash or Ctrl-C never loses more than ~17 minutes.

## Repository layout

```
mod/                    C# mod source
  BridgeServer.cs         TCP server, one JSON message per line
  EpisodeManager.cs       lockstep action/state cycle + reset macro (boot → bench → statue → fight)
  StateReader.cs          reads Knight/Hornet state from the live game
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
    dashboard.py            serve the web dashboard (default port 9021)
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

## Setup

**1. Build and install the mod:**

```bash
bash mod/build.sh                                      # macOS
powershell -ExecutionPolicy Bypass -File mod\build.ps1 # Windows
```

Both assume the default Steam location; for a different Steam library pass
`-HKManaged <path>` to build.ps1 or edit `HK_MANAGED` in build.sh.

**2. Re-sign the game — macOS only** (required after *every* mod build — copying the DLL into the app bundle invalidates its code signature, and unsigned launches die instantly with exit code 138). Windows has no code-signing step:

```bash
codesign --force --deep --sign - \
  "$HOME/Library/Application Support/Steam/steamapps/common/Hollow Knight/hollow_knight.app"
```

**3. Set up the Python environment:**

```bash
cd trainer
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

**4. Run the test suite** (no game needed — tests run against a scripted fake game):

```bash
cd trainer && ./.venv/bin/python -m pytest -q
```

**5. One-time game prep:** the training save file should be parked in Godhome with Hornet 1 unlocked in the Hall of Gods, ideally resting at the Hall of Gods bench. The mod's boot macro handles everything from the title screen onward.

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

`train.py` launches the game itself — don't start one manually. When prompted, wait for the game to reach the Hall of Gods (the boot macro can drive it there; a few `reset ... reconnecting` retries on stderr are normal) and press Enter. 500k steps is roughly an overnight run (~54k steps/hour at 15&nbsp;Hz).

Useful flags: `--gen-every` (checkpoint interval, default 15000), `--n-steps` / `--batch-size` / `--n-epochs` (PPO rollout/update shape), `--seed`, `--root` (default `~/hkrl`).

### Monitor progress

The web dashboard shows every run under `~/hkrl/runs/`: live/stopped status,
timestep progress and ETA, steps/hour, the learning curves (boss damage, win
rate, reward, episode length per generation), and a per-episode reward chart
that updates between checkpoints. It is read-only — it never touches the game
port — so it is safe to leave up beside a live run:

```bash
./.venv/bin/python scripts/dashboard.py --open   # http://127.0.0.1:9021
```

Each run lives in `~/hkrl/runs/<run-id>/`. The per-generation manifest is the health record:

```bash
tail -f ~/hkrl/runs/my-run/generations.jsonl
```

Healthy: `episodes` > 0 each generation, `mean_boss_damage` trending up, `recoveries` rare. Warning signs: `episodes: 0`, `recoveries` climbing every generation (App Nap — check window occlusion), or episode length pinned at 2700 with zero damage. TensorBoard logs are in the run's `tb/` directory:

```bash
./.venv/bin/tensorboard --logdir ~/hkrl/runs/my-run/tb
```

### Pause and resume

**Pause:** press Ctrl-C once. The run finishes the current episode (≤ ~3 minutes), saves a final generation, and shuts the game down. A second Ctrl-C forces an immediate abort (it still attempts a save).

**Resume:**

```bash
caffeinate -dims ./.venv/bin/python scripts/train.py --resume ~/hkrl/runs/my-run
```

This relaunches the game, reloads the latest generation's weights **and** its observation-normalization statistics, and continues the same run — same directory, continued generation numbering and TensorBoard curves. `--timesteps` is additive on resume ("collect this many more"). If a run dies overnight (recovery exhausted), it prints this exact `--resume` command before exiting; PPO hyperparameters come from the checkpoint on resume, so the `--n-steps`/`--batch-size`/`--n-epochs` flags only shape fresh runs.

### Watch a checkpoint play

After training stops (never against a port a live trainer is using), launch a game instance and replay any generation:

```bash
./.venv/bin/python scripts/launch_instances.py
./.venv/bin/python scripts/replay.py --run-dir ~/hkrl/runs/my-run --episodes 3          # latest gen
./.venv/bin/python scripts/replay.py --run-dir ~/hkrl/runs/my-run --gen 1 --episodes 3  # a specific gen
```

Comparing an early generation against the latest is the quickest way to *see* what it has learned.

## Things that bite

- **Forgot to re-sign after a mod build (macOS)** → game exits immediately with code 138. Run the `codesign` command above.
- **Occluded game window (macOS) / minimized window (Windows)** → the OS suspends or deprioritizes the game; the trainer sees a wedge and burns a recovery. Keep it visible, even if small.
- **Old checkpoints after changing the action/observation space** → not resumable or replayable; the policy network's shape changed. Start a fresh run.
- **The mod drops idle connections after 10 s** → anything that blocks the trainer between messages for ~10 s (including a slow PPO update) severs the connection. If updates get slow, lower `--n-epochs` first; don't raise the mod's `ReadTimeout`.
