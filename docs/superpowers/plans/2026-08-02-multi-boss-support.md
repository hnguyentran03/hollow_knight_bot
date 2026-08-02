# Multi-Boss Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the boss a configuration axis (trainer + mod + dashboard), validated by training a policy to reliably beat Gruz Mother.

**Architecture:** Split registries keyed by a shared boss id: `trainer/hkrl/bosses.py` holds trainer-side data (FSM state list, arena constants), a new `mod/BossRegistry.cs` holds mod-side data (scene, boss GameObject name, statue X, HP ceiling, tier index, needle name). The trainer sends `"boss": "<id>"` in every reset request; protocol version bumps 1→2 so a new trainer refuses an old mod. Spec: `docs/superpowers/specs/2026-08-02-multi-boss-support-design.md`.

**Tech Stack:** Python 3 (gymnasium, SB3/sb3-contrib, pytest), C# Unity mod (Modding API, Newtonsoft.Json).

## Global Constraints

- Run Python tests as `.venv/bin/python -m pytest` **from the `trainer/` directory** (bare `pytest` cannot import `hkrl`).
- Build the mod with `mod/build.sh` (macOS); a build that compiles is the C# verification bar — there are no C# unit tests.
- Commit messages: 1–3 plain sentences, high-level, **no** conventional-commit prefixes (no "feat:", "fix:", etc.).
- Default boss is `hornet1` everywhere; every existing behavior at `hornet1` must be unchanged (existing tests keep passing).
- Protocol version is 2 after this plan; the constant lives in `trainer/hkrl/protocol.py` as `PROTOCOL_VERSION = 2` and in `mod/BridgeServer.cs`'s hello line.
- Tasks 1–9 must not require the real game. Tasks 10–12 are real-game phases (user present).

---

### Task 1: Trainer boss registry (`bosses.py`)

**Files:**
- Create: `trainer/hkrl/bosses.py`
- Modify: `trainer/hkrl/env.py` (move Hornet constants out)
- Test: `trainer/tests/test_bosses.py` (create)

**Interfaces:**
- Produces: `BossSpec` frozen dataclass with fields `id: str`, `fsm_states: tuple[str, ...]`, `arena_center_x: float`, `arena_half_w: float`, `floor_y: float`, `arena_height: float`; `BOSSES: dict[str, BossSpec]` containing `"hornet1"`; `get_boss(boss_id: str) -> BossSpec` raising `ValueError` (naming the known ids) on unknown id.
- Consumes: the constant values currently at `trainer/hkrl/env.py:17-45` (`HORNET_STATES`, `ARENA_CENTER_X = 26.5`, `ARENA_HALF_W = 11.23`, `FLOOR_Y = 28.41`, `ARENA_HEIGHT = 9.59`).

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_bosses.py`:

```python
import pytest

from hkrl.bosses import BOSSES, get_boss


def test_hornet1_spec_matches_the_measured_constants():
    # The values recorded in mod/DISCOVERED.md sections 1-2; moving them into
    # the registry must not change them.
    spec = get_boss("hornet1")
    assert spec.id == "hornet1"
    assert len(spec.fsm_states) == 28          # 27 recorded states + UNKNOWN
    assert spec.fsm_states[-1] == "UNKNOWN"
    assert spec.arena_center_x == 26.5
    assert spec.arena_half_w == 11.23
    assert spec.floor_y == 28.41
    assert spec.arena_height == 9.59


def test_get_boss_rejects_unknown_ids_naming_the_known_ones():
    with pytest.raises(ValueError, match="hornet1"):
        get_boss("grimm")


def test_registry_keys_match_spec_ids():
    assert all(spec.id == key for key, spec in BOSSES.items())
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `trainer/`): `.venv/bin/python -m pytest tests/test_bosses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hkrl.bosses'`

- [ ] **Step 3: Create `trainer/hkrl/bosses.py`**

```python
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
}


def get_boss(boss_id: str) -> BossSpec:
    try:
        return BOSSES[boss_id]
    except KeyError:
        known = ", ".join(sorted(BOSSES))
        raise ValueError(
            f"unknown boss {boss_id!r}; known bosses: {known}") from None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bosses.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add trainer/hkrl/bosses.py trainer/tests/test_bosses.py
git commit -m "Add a trainer-side boss registry: a BossSpec per boss id holding the FSM state list and arena constants the observation space is built from, starting with hornet1's measured values."
```

---

### Task 2: `HKEnv` takes a boss; obs space built from the spec

**Files:**
- Modify: `trainer/hkrl/env.py`
- Test: `trainer/tests/test_env.py`

**Interfaces:**
- Consumes: `get_boss` from Task 1.
- Produces: `HKEnv(boss="hornet1", ...)` keyword; `self.boss: BossSpec` attribute. `HORNET_STATES`/`ARENA_*`/`FLOOR_Y` module constants are **removed** from `env.py` (nothing else imports them — `random_agent.py` imports only `HKEnv` and `OBS_KEYS`, which stays).

- [ ] **Step 1: Write the failing tests**

Append to `trainer/tests/test_env.py`:

```python
def test_env_rejects_an_unknown_boss_before_connecting():
    # No FakeGame: the registry lookup must fail before any socket work.
    with pytest.raises(ValueError, match="hornet1"):
        HKEnv(port=1, boss="grimm")


def test_obs_size_is_scalar_block_plus_boss_state_onehot():
    from hkrl.bosses import get_boss
    from hkrl.env import OBS_KEYS
    episode = [state(obs())]
    with FakeGame([episode]) as fg:
        env = HKEnv(port=fg.port)   # default boss: hornet1
        n = len(OBS_KEYS) + len(get_boss("hornet1").fsm_states)
        assert env.observation_space.shape == (n,)
        env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_env.py -v -k "unknown_boss or onehot"`
Expected: `test_env_rejects_an_unknown_boss_before_connecting` FAILS (`TypeError: __init__() got an unexpected keyword argument 'boss'`); the obs-size test passes already (46 = 18 + 28) — it pins the invariant for later bosses.

- [ ] **Step 3: Rework `env.py`**

In `trainer/hkrl/env.py`:

1. Delete the module-level block at lines 13–45 (`HORNET_STATES`, the arena-constants comment block, `ARENA_CENTER_X`, `ARENA_HALF_W`, `FLOOR_Y`, `ARENA_HEIGHT`). Keep `VEL_SCALE = 20.0` (boss-independent). Add the import:

```python
from hkrl.bosses import get_boss
```

2. In `__init__`, add the keyword and resolve it FIRST (before the socket connect), and size the one-hot from it:

```python
    def __init__(self, host="127.0.0.1", port=9020, reward_config=None,
                 max_steps=2700, timeout=30.0, reset_retries=8,
                 keepalive=3.0, reset_log_dir=None, boss="hornet1"):
        # Resolved before any socket work so an unknown id fails instantly
        # and locally, not after a game connection is already up.
        self.boss = get_boss(boss)
```

and replace `n = len(OBS_KEYS) + len(HORNET_STATES)` with:

```python
        n = len(OBS_KEYS) + len(self.boss.fsm_states)
```

3. In `_flatten`, replace every use of the deleted constants with the spec (the needle terms keep the same scales; a boss without a needle always reports `needle_active` false, so those slots read zero):

```python
        b = self.boss
        v = [
            (obs["kx"] - b.arena_center_x) / b.arena_half_w,
            (obs["ky"] - b.floor_y) / b.arena_height,
            obs["kvx"] / VEL_SCALE, obs["kvy"] / VEL_SCALE,
            obs["khp"] / 9.0, obs["soul"] / 99.0,
            float(obs["on_ground"]), float(obs["dashing"]),
            float(obs["invuln"]), float(obs["facing_right"]),
            (obs["bx"] - obs["kx"]) / b.arena_half_w,
            (obs["by"] - obs["ky"]) / b.arena_height,
            obs["bvx"] / VEL_SCALE, obs["bvy"] / VEL_SCALE,
            obs["bhp"] / self._max_bhp,
            float(obs["needle_active"]),
            (obs["nx"] - obs["kx"]) / b.arena_half_w if obs["needle_active"] else 0.0,
            (obs["ny"] - obs["ky"]) / b.arena_height if obs["needle_active"] else 0.0,
        ]
        onehot = [0.0] * len(b.fsm_states)
        try:
            onehot[b.fsm_states.index(obs["boss_state"])] = 1.0
        except ValueError:
            onehot[-1] = 1.0
```

- [ ] **Step 4: Run the whole env test file**

Run: `.venv/bin/python -m pytest tests/test_env.py -v`
Expected: all PASS (the moved constants change no values).

- [ ] **Step 5: Commit**

```bash
git add trainer/hkrl/env.py trainer/tests/test_env.py
git commit -m "HKEnv takes a boss id and builds its observation space and normalization from that boss's registry spec, making policies boss-specific by construction. Hornet's constants now live only in the registry."
```

---

### Task 3: Protocol v2 — boss id in reset, version gate, error replies

**Files:**
- Modify: `trainer/hkrl/protocol.py`, `trainer/hkrl/env.py`, `trainer/hkrl/fake_game.py`
- Test: `trainer/tests/test_env.py`

**Interfaces:**
- Produces: `PROTOCOL_VERSION = 2` in `hkrl.protocol`; `HKEnv` sends `{"type": "reset", "boss": <id>}` and raises `RuntimeError` on (a) a hello whose `version != PROTOCOL_VERSION`, (b) an `{"type": "error", "message": ...}` reply to reset. `FakeGame(version=2, bosses=("hornet1", "gruz_mother"))` keywords; `FakeGame.reset_bosses: list` records the boss id of every reset request; an unknown boss id gets `{"type": "error", ...}` then a connection drop.
- Consumes: `HKEnv.reset` retry loop at `trainer/hkrl/env.py:205-241`; `FakeGame._serve` at `trainer/hkrl/fake_game.py:91-126`.

- [ ] **Step 1: Write the failing tests**

Append to `trainer/tests/test_env.py`:

```python
def test_reset_sends_the_boss_id():
    with FakeGame([[state(obs())]]) as fg:
        env = HKEnv(port=fg.port)
        env.reset()
        assert fg.reset_bosses == ["hornet1"]
        env.close()


def test_old_mod_version_is_refused_at_connect():
    with FakeGame([[state(obs())]], version=1) as fg:
        with pytest.raises(RuntimeError, match="protocol"):
            HKEnv(port=fg.port)


def test_mod_error_reply_fails_the_reset_loudly():
    # A mod that doesn't know the requested boss answers with an error
    # instead of a state; that must raise, not retry or hang.
    with FakeGame([[state(obs())]], bosses=("gruz_mother",)) as fg:
        env = HKEnv(port=fg.port)
        with pytest.raises(RuntimeError, match="hornet1"):
            env.reset()
        env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_env.py -v -k "boss_id or old_mod or error_reply"`
Expected: FAIL (`reset_bosses` missing; `version`/`bosses` unexpected keywords).

- [ ] **Step 3: Implement**

In `trainer/hkrl/protocol.py`, update the module docstring to say `(protocol v2)` and add below the imports:

```python
# The version the mod must greet with ({"type": "hello", "version": N}).
# Bumped 1 -> 2 when the reset request gained a required "boss" id: a v1 mod
# would ignore the field and silently fight Hornet while the trainer builds
# a different boss's observation space -- exactly the mismatch a version
# check exists to catch. The consumer-side check lives in HKEnv.__init__.
PROTOCOL_VERSION = 2
```

In `trainer/hkrl/env.py`:

1. Import it: `from hkrl.protocol import Connection, ConnectionClosed, PROTOCOL_VERSION`
2. In `__init__`, right after `self.conn.connect()`:

```python
        version = (self.conn.hello or {}).get("version")
        if version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"mod speaks protocol v{version}, this trainer needs "
                f"v{PROTOCOL_VERSION} -- rebuild the mod (mod/build.sh) and "
                f"restart the game")
```

3. In `reset()`, send the boss id and handle an error reply (inside the retry loop, replacing the bare `send`/`recv`/`break`):

```python
                self.conn.send({"type": "reset", "boss": self.boss.id})
                msg = self.conn.recv()
                if msg.get("type") == "error":
                    # The mod refused the reset (e.g. a boss id its registry
                    # doesn't know). A registry-skew bug, not protocol
                    # rhythm: fail loudly instead of retrying into the same
                    # refusal.
                    raise RuntimeError(
                        f"mod refused reset: {msg.get('message', msg)}")
                break
```

In `trainer/hkrl/fake_game.py`:

1. Extend `__init__` signature to `def __init__(self, episodes, port=0, fail_resets=0, hang_resets=0, version=2, bosses=("hornet1", "gruz_mother")):` and add:

```python
        # Protocol version greeted on connect; tests pass 1 to simulate a
        # stale mod build.
        self.version = version
        # Boss ids this fake's "mod registry" knows; a reset naming any
        # other id is answered with an error then a drop, like the real mod.
        self.bosses = tuple(bosses)
        # Boss id carried by each reset request, for assertions.
        self.reset_bosses = []
```

2. In `_serve`, greet with the configured version: `send({"type": "hello", "version": self.version})`
3. In the `reset` branch, before the `fail_resets` handling, record and validate:

```python
                if msg["type"] == "reset":
                    boss = msg.get("boss")
                    self.reset_bosses.append(boss)
                    if boss not in self.bosses:
                        send({"type": "error",
                              "message": f"unknown boss {boss!r}"})
                        conn.shutdown(socket.SHUT_RDWR)
                        return
```

- [ ] **Step 4: Run the full trainer suite**

Run: `.venv/bin/python -m pytest`
Expected: all PASS (every existing FakeGame consumer sends `hornet1`, which the default `bosses` accepts).

- [ ] **Step 5: Commit**

```bash
git add trainer/hkrl/protocol.py trainer/hkrl/env.py trainer/hkrl/fake_game.py trainer/tests/test_env.py
git commit -m "Bump the bridge protocol to v2: every reset names its boss id, the trainer refuses a v1 hello at connect, and a mod error reply fails the reset loudly instead of retrying. The fake game validates boss ids the same way."
```

---

### Task 4: `train.py --boss` with resume guard and config recording

**Files:**
- Modify: `trainer/scripts/train.py`, `README.md`
- Test: `trainer/tests/test_train.py`

**Interfaces:**
- Produces: `resolve_boss(flag: str | None, run_dir: Path | None) -> str` in `train.py` — `run_dir=None` means a fresh run (returns `flag or "hornet1"`); a resume reads the run's recorded boss (`"hornet1"` when the config predates the field) and raises `ValueError` if an explicit flag conflicts. `--boss` argparse flag (default `None`, `choices=sorted(BOSSES)`); resolved value recorded in `config.jsonl` and passed to the env stack as `boss=<id>`.
- Consumes: `get_boss`/`BOSSES` (Task 1); `read_jsonl` from `hkrl.rundata`; `build_env`'s `**supervisor_kwargs` pass-through (`trainer/scripts/train.py:107`) which lands in `SupervisedVecEnv`'s `env_kwargs` and reaches every worker `HKEnv`.

- [ ] **Step 1: Write the failing tests**

Append to `trainer/tests/test_train.py` (match its existing import style for the train module):

```python
def test_resolve_boss_fresh_run_defaults_to_hornet1():
    assert train.resolve_boss(None, None) == "hornet1"
    assert train.resolve_boss("gruz_mother", None) == "gruz_mother"


def test_resolve_boss_resume_reads_the_recorded_boss(tmp_path):
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"boss": "gruz_mother"}) + "\n")
    assert train.resolve_boss(None, tmp_path) == "gruz_mother"


def test_resolve_boss_resume_without_a_recorded_boss_is_hornet1(tmp_path):
    # Runs from before the boss field existed.
    (tmp_path / "config.jsonl").write_text(json.dumps({"instances": 1}) + "\n")
    assert train.resolve_boss(None, tmp_path) == "hornet1"


def test_resolve_boss_refuses_a_conflicting_flag_on_resume(tmp_path):
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"boss": "hornet1"}) + "\n")
    with pytest.raises(ValueError, match="gruz_mother"):
        train.resolve_boss("gruz_mother", tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train.py -v -k resolve_boss`
Expected: FAIL with `AttributeError: ... has no attribute 'resolve_boss'`

- [ ] **Step 3: Implement in `train.py`**

1. Imports: add `from hkrl.bosses import BOSSES` and `from hkrl.rundata import read_jsonl`.
2. Add near `resolve_async_resets`:

```python
def resolve_boss(flag: str | None, run_dir: Path | None) -> str:
    """The boss this session fights. Fresh runs take the flag (default
    hornet1). A resume takes the run's recorded boss -- the checkpoint's
    observation space is built from it, so it is not overridable: an
    explicit conflicting --boss is a hard error here, with a clear message
    instead of a shape mismatch deep inside model load. Configs from before
    the boss field read as hornet1."""
    if run_dir is None:
        return flag or "hornet1"
    configs = read_jsonl(run_dir / "config.jsonl")
    recorded = (configs[-1].get("boss") if configs else None) or "hornet1"
    if flag is not None and flag != recorded:
        raise ValueError(
            f"--boss {flag} conflicts with {run_dir}'s recorded boss "
            f"{recorded!r}; a checkpoint's observation space is built for "
            f"its boss, so a resume always keeps it. Start a new run to "
            f"train against {flag}.")
    return recorded
```

3. Add the flag after `--target-kl`:

```python
    ap.add_argument("--boss", default=None, choices=sorted(BOSSES),
                    help="which boss to train against (default: hornet1). "
                         "Sets the observation space, so checkpoints are "
                         "boss-specific: a resume always keeps the run's "
                         "recorded boss and refuses a conflicting flag.")
```

4. In `main()`, right after the `resume`/`run_dir` block (after line 389, before the config dump so the resolved value is recorded):

```python
    try:
        args.boss = resolve_boss(args.boss,
                                 run_dir if args.resume is not None else None)
    except ValueError as exc:
        sys.exit(str(exc))
```

(`build_config_dict` serializes `vars(args)`, so the resolved `args.boss` lands in `config.jsonl` with no further change.)

5. In the `build_env(...)` call, add the pass-through kwarg (it rides `**supervisor_kwargs` into every worker's `HKEnv`):

```python
            boss=args.boss,
```

6. `README.md`: in the section documenting `--target-kl`, add a short `--boss` paragraph: which bosses exist (`hornet1` default, `gruz_mother`), that checkpoints are boss-specific, and that a resume keeps the run's recorded boss and refuses a conflicting flag.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_train.py tests/test_env.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/scripts/train.py trainer/tests/test_train.py README.md
git commit -m "Add a --boss flag to train.py: the resolved boss lands in config.jsonl and flows to every worker env. A resume always keeps the run's recorded boss, refusing a conflicting flag up front rather than failing on observation shapes at model load."
```

---

### Task 5: Replay reads the run's boss

**Files:**
- Modify: `trainer/scripts/replay.py`
- Test: `trainer/tests/test_replay.py`

**Interfaces:**
- Consumes: `make_env(port, host=host)` call at `trainer/scripts/replay.py:54`; `read_jsonl` from `hkrl.rundata`; the `boss` env kwarg (Task 2).
- Produces: replay builds its env with the boss recorded in the run's `config.jsonl` (default `"hornet1"` when absent). Extract the lookup as `run_boss(run_dir) -> str` so it is testable without a game.

- [ ] **Step 1: Write the failing test**

Append to `trainer/tests/test_replay.py` (match its existing imports; it already imports the replay script module):

```python
def test_run_boss_reads_config_and_defaults_to_hornet1(tmp_path):
    assert replay.run_boss(tmp_path) == "hornet1"          # no config at all
    (tmp_path / "config.jsonl").write_text(
        json.dumps({"boss": "gruz_mother"}) + "\n")
    assert replay.run_boss(tmp_path) == "gruz_mother"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_replay.py -v -k run_boss`
Expected: FAIL with `AttributeError: ... no attribute 'run_boss'`

- [ ] **Step 3: Implement**

In `trainer/scripts/replay.py`, add (importing `read_jsonl` from `hkrl.rundata` if not already imported):

```python
def run_boss(run_dir) -> str:
    """The boss the run trained against, from its recorded config; runs
    from before the boss field read as hornet1. The checkpoint's
    observation space was built for this boss, so the replay env must be
    too."""
    configs = read_jsonl(Path(run_dir) / "config.jsonl")
    return ((configs[-1].get("boss") if configs else None) or "hornet1")
```

and thread it into the env construction at line 54 (the function this line sits in has the run dir in scope; pass it through as a parameter if not):

```python
    venv = DummyVecEnv([make_env(port, host=host, boss=run_boss(run_dir))])
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_replay.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/scripts/replay.py trainer/tests/test_replay.py
git commit -m "Replay builds its env for the boss the run recorded in config.jsonl, so replaying a run trained against a non-default boss reconstructs the matching observation space."
```

---

### Task 6: Launcher knows `boss`

**Files:**
- Modify: `trainer/hkrl/launcher.py`
- Test: `trainer/tests/test_launcher_module.py`

**Interfaces:**
- Consumes: `_validate` / `command` / `_restart_params` / `_INT_PARAMS` in `trainer/hkrl/launcher.py:80-210`; `BOSSES` from Task 1.
- Produces: `boss` accepted in launch params — validated against the registry, forwarded as `--boss <id>` on **new** runs only (a resume derives it from the run's config, per Task 4), and carried through `_restart_params` so a checkpoint-less restart keeps its boss.

- [ ] **Step 1: Write the failing tests**

Append to `trainer/tests/test_launcher_module.py` (match its existing style — it calls `launcher.command(root, params)` and asserts on the argv):

```python
def test_command_forwards_boss_on_new_runs(tmp_path):
    cmd = launcher.command(tmp_path, {"mode": "new", "run_id": "r1",
                                      "boss": "gruz_mother"},
                           platform="linux")
    assert "--boss" in cmd
    assert cmd[cmd.index("--boss") + 1] == "gruz_mother"


def test_command_drops_boss_on_resume(tmp_path):
    # A resume derives the boss from the run's own config (train.py's
    # resolve_boss); forwarding it would be redundant at best.
    cmd = launcher.command(tmp_path, {"mode": "resume", "run_id": "r1",
                                      "boss": "gruz_mother"},
                           platform="linux")
    assert "--boss" not in cmd


def test_validate_rejects_an_unknown_boss(tmp_path):
    with pytest.raises(ValueError, match="boss"):
        launcher.command(tmp_path, {"mode": "new", "run_id": "r1",
                                    "boss": "grimm"}, platform="linux")


def test_restart_params_carries_the_boss(tmp_path):
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "config.jsonl").write_text(
        json.dumps({"boss": "gruz_mother", "n_steps": 512}) + "\n")
    params = launcher._restart_params(run_dir, {"mode": "resume",
                                                "run_id": "r1"})
    assert params["boss"] == "gruz_mother"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_launcher_module.py -v -k boss`
Expected: the forward/carry tests FAIL (no `--boss` in argv, no `boss` key); the unknown-boss test FAILS (no ValueError).

- [ ] **Step 3: Implement in `launcher.py`**

1. Import: `from hkrl.bosses import BOSSES`.
2. Next to the param tuples (line 145-147):

```python
# String-valued params. boss is new-only for the same reason the model-
# shaping ints are: on resume train.py derives it from the run's recorded
# config (and refuses a conflicting flag), so forwarding it is noise.
_STR_NEW_ONLY = ("boss",)
```

3. In `_validate`, after the int-param loop:

```python
    boss = params.get("boss")
    if boss not in (None, ""):
        if boss not in BOSSES:
            raise ValueError(
                f"unknown boss {boss!r}; known: {', '.join(sorted(BOSSES))}")
        clean["boss"] = boss
```

4. In `command`, extend the forwarding loop's key source for new mode:

```python
    keys = _ALWAYS if p["mode"] == "resume" else _INT_PARAMS + _STR_NEW_ONLY
    for key in keys:
        if key in p:
            cmd += ["--" + key.replace("_", "-"), str(p[key])]
```

5. In `_restart_params`, widen the copied keys: `for key in _INT_PARAMS + _STR_NEW_ONLY:`

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_launcher_module.py tests/test_launcher.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/hkrl/launcher.py trainer/tests/test_launcher_module.py
git commit -m "Teach the launcher a boss parameter: validated against the registry, forwarded to train.py on new runs only, and carried through checkpoint-less restarts."
```

---

### Task 7: Dashboard boss picker and per-run boss display

**Files:**
- Modify: `trainer/hkrl/dashboard.py`, `trainer/hkrl/dashboard.html`
- Test: `trainer/tests/test_dashboard.py`

**Interfaces:**
- Consumes: `/api/launcher` GET handler (`trainer/hkrl/dashboard.py:49-52`); launch form (`trainer/hkrl/dashboard.html:333-344`); form-collection JS (`dashboard.html:906` iterates `#launch-form input`); run-status meta (`dashboard.html:759`), config-detail keys (`dashboard.html:823`), previous-runs row (`dashboard.html:1071`); `BOSSES` from Task 1.
- Produces: `/api/launcher` response gains `"bosses": sorted(BOSSES)`; the launch form gains a `<select name="boss">` populated from it; run cards and the config details show the run's boss (`hornet1` fallback).

- [ ] **Step 1: Write the failing test**

Append to `trainer/tests/test_dashboard.py` (match its existing server-fixture style for GET assertions):

```python
def test_api_launcher_lists_the_known_bosses(dashboard_server):
    body = get_json(dashboard_server, "/api/launcher")
    assert body["bosses"] == sorted(hkrl.bosses.BOSSES)
```

(Adapt fixture/helper names to the file's existing ones when writing it in — the file already has `/api/launcher` tests to copy from.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -v -k bosses`
Expected: FAIL with `KeyError: 'bosses'`

- [ ] **Step 3: Implement**

1. `dashboard.py`: import `from hkrl.bosses import BOSSES`; in the `/api/launcher` response dict add `"bosses": sorted(BOSSES),`.
2. `dashboard.html` form: after the Instances row (line 335), add:

```html
      <div><label for="f-boss">Boss</label><select id="f-boss" name="boss"></select></div>
```

3. JS: where the launcher poll handles the `/api/launcher` payload, populate the select once (only when its option set differs, mirroring the run-select guard at line 850):

```javascript
  const bossSel = document.getElementById("f-boss");
  const bosses = data.bosses || ["hornet1"];
  if ([...bossSel.options].map(o => o.value).join() !== bosses.join()) {
    bossSel.replaceChildren();
    for (const b of bosses) {
      const opt = document.createElement("option");
      opt.value = b; opt.textContent = b;
      bossSel.appendChild(opt);
    }
    bossSel.value = "hornet1";
  }
```

4. JS form collection (line 906): change the selector to `"#launch-form input, #launch-form select"` so the boss reaches the POST body.
5. Run display: in the run-status meta (near line 759) add the boss beside instances, reading `run.config && run.config.boss ? run.config.boss : "hornet1"`; add `"boss"` to the config-detail `keys` array (line 823); in the previous-runs row (line 1071) prepend the boss the same way instances are shown, with the same `hornet1` fallback for pre-field runs.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add trainer/hkrl/dashboard.py trainer/hkrl/dashboard.html trainer/tests/test_dashboard.py
git commit -m "Add a boss picker to the dashboard launch form, fed by the registry via /api/launcher, and show each run's boss on its cards and config details with a hornet1 fallback for older runs."
```

---

### Task 8: Mod boss registry + protocol v2 (hello, boss parsing, error reply)

**Files:**
- Create: `mod/BossRegistry.cs`
- Modify: `mod/BridgeServer.cs`, `mod/EpisodeManager.cs`

**Interfaces:**
- Produces: `BossSpec` (C#) with `Id`, `Scene`, `ObjectName`, `StatueX`, `MaxAttunedHp`, `TierIndex`, `NeedleName` (nullable); `BossRegistry.All` (dict), `BossRegistry.Current` (defaults to `hornet1`), `BossRegistry.TrySet(string id)`; `BridgeServer.SendError(string message)`; hello line says `"version":2`.
- Consumes: reset-acceptance sites `mod/EpisodeManager.cs:198-204` (idle) and `246-253` (mid-episode); hello write `mod/BridgeServer.cs:77`.

- [ ] **Step 1: Create `mod/BossRegistry.cs`**

```csharp
// mod/BossRegistry.cs
using System.Collections.Generic;

namespace HKRLBot
{
    // Mod-side per-boss data, keyed by the boss id the trainer sends in
    // every reset request (protocol v2). The trainer keeps its own registry
    // (trainer/hkrl/bosses.py) holding the data IT consumes (FSM states,
    // arena constants); this one holds only what the mod consumes. Adding a
    // boss means one entry in each, transcribed from an in-game discovery
    // session recorded in DISCOVERED.md.
    public class BossSpec
    {
        public string Id;
        // The Godhome arena scene this boss's fight runs in.
        public string Scene;
        // The boss's root GameObject name, for StateReader's scene scan.
        public string ObjectName;
        // Knight X when standing at this boss's statue in GG_Workshop --
        // a measured value (F1 overlay), like Hornet's in DISCOVERED.md
        // section 3. A wrong-but-plausible number silently corrupts the
        // reset macro; never change without re-measuring.
        public float StatueX;
        // Ceiling for backstop B (wrong-difficulty detection): safely above
        // the boss's measured Attuned max HP, below its next tier's.
        public int MaxAttunedHp;
        // BossChallengeUI.LoadBoss(index) tier index for Attuned. 0 on the
        // Hornet statue (DISCOVERED.md section 5); re-verify per statue.
        public int TierIndex;
        // GameObject name of a boss-owned projectile worth tracking in the
        // observation (Hornet's thrown needle), or null when the boss has
        // none -- the needle obs fields then always read inactive.
        public string NeedleName;
    }

    public static class BossRegistry
    {
        public static readonly Dictionary<string, BossSpec> All =
            new Dictionary<string, BossSpec>
            {
                ["hornet1"] = new BossSpec
                {
                    Id = "hornet1",
                    Scene = "GG_Hornet_1",
                    ObjectName = "Hornet Boss 1",
                    StatueX = 62.21f,
                    MaxAttunedHp = 1000,
                    TierIndex = 0,
                    NeedleName = "Needle",
                },
            };

        // The boss the current/next episode fights. Defaults to hornet1 so
        // pre-reset code paths (overlay, early reads) always have a spec;
        // set by EpisodeManager whenever a reset message is accepted.
        public static BossSpec Current = All["hornet1"];

        public static bool TrySet(string id)
        {
            if (id != null && All.TryGetValue(id, out var spec))
            {
                Current = spec;
                return true;
            }
            return false;
        }
    }
}
```

- [ ] **Step 2: Bump the hello and add SendError in `BridgeServer.cs`**

Line 77 becomes:

```csharp
                    writer.WriteLine("{\"type\":\"hello\",\"version\":2}");
```

Add beside `SendPong` (same gated, IOException-drops write path):

```csharp
        // Refuse a request with a reason the trainer can surface (e.g. a
        // boss id this build's registry doesn't know). Same bounded, gated
        // write path as SendState/SendPong; the caller decides whether to
        // Drop() afterward.
        public void SendError(string message)
        {
            var msg = new JObject { ["type"] = "error", ["message"] = message };
            var text = msg.ToString(Formatting.None);
            lock (gate)
            {
                try { writer?.WriteLine(text); }
                catch (IOException) { DropLocked(); }
            }
        }
```

- [ ] **Step 3: Parse the boss id at both reset-acceptance sites in `EpisodeManager.cs`**

Add a private helper to `EpisodeManager`:

```csharp
        // Protocol v2: every reset names its boss. An unknown id is a
        // registry-skew bug (trainer knows a boss this build doesn't) --
        // refuse with an error the trainer raises on, rather than silently
        // fighting whatever Current already was. Returns false when refused.
        private bool TrySetBossFromReset(BridgeServer server, JObject msg)
        {
            string id = (string)msg["boss"];
            if (BossRegistry.TrySet(id)) return true;
            HKRLBotMod.Instance.Log(
                $"EpisodeManager: reset named unknown boss '{id}' -- refusing "
                + "and dropping the connection (rebuild skew between trainer "
                + "and mod registries?)");
            server.SendError($"unknown boss '{id}'; this mod build knows: "
                + string.Join(", ", BossRegistry.All.Keys));
            server.Drop();
            return false;
        }
```

At the idle-branch acceptance (line 198), guard before flipping any state:

```csharp
                if ((string)msg["type"] == "reset")
                {
                    if (!TrySetBossFromReset(server, msg)) return;
                    awaitingReset = true;
                    ...
```

At the mid-episode acceptance (`case "reset":`, line 246), the refusal must also end the episode (the connection is gone):

```csharp
                    case "reset":
                        if (!TrySetBossFromReset(server, reply))
                        {
                            episodeActive = false;
                            HKRLBotMod.Instance.Input.Clear();
                            break;
                        }
                        episodeActive = false;
                        awaitingReset = true;
                        ...
```

- [ ] **Step 4: Build**

Run: `mod/build.sh`
Expected: build succeeds. (No C# tests; the trainer-side FakeGame tests from Task 3 pin the protocol shape both sides now implement.)

- [ ] **Step 5: Commit**

```bash
git add mod/BossRegistry.cs mod/BridgeServer.cs mod/EpisodeManager.cs
git commit -m "Add the mod-side boss registry and speak protocol v2: the hello advertises version 2, every accepted reset sets the current boss from its id, and an unknown id is refused with an error reply instead of silently fighting the previous boss."
```

---

### Task 9: Parameterize the mod's boss-specific code paths

**Files:**
- Modify: `mod/EpisodeManager.cs`, `mod/StateReader.cs`

**Interfaces:**
- Consumes: `BossRegistry.Current` (Task 8).
- Produces: no hardcoded Hornet facts left outside `BossRegistry.All`. Specifically replaced: `BossScene` const (`EpisodeManager.cs:18`), `MaxAttunedHornetHp` (`:100`), `ResetMacro.StatueX` (`:802`), the `scene == "GG_Hornet_1"` macro branch (`:987`), `GameObject.Find("Hornet Boss 1")` (`StateReader.cs:110`), `GameObject.Find("Needle")` (`:123`), `LoadBoss` invoke arg `0` (`:271`).

- [ ] **Step 1: `EpisodeManager.cs`**

1. Replace the const at line 18 with a property (every existing `BossScene` use site — the scene-change latch, `done` checks, `TickReset`'s live check — then follows the current boss with no further edits):

```csharp
        private static string BossScene => BossRegistry.Current.Scene;
```

2. Delete the `MaxAttunedHornetHp` const (lines 94-100) and change its use in `TickReset` (line 500-504) to:

```csharp
                if (b.Hp > BossRegistry.Current.MaxAttunedHp)
                {
                    HKRLBotMod.Instance.Log(
                        $"EpisodeManager: fight went live at bossMaxHp={b.Hp}, above the "
                        + $"Attuned ceiling for boss '{BossRegistry.Current.Id}' "
                        + $"({BossRegistry.Current.MaxAttunedHp}) -- wrong difficulty "
                        + "tier. Clearing input and dropping the connection.");
```

3. In `ResetMacro`: delete the `StatueX` const (line 802) and add `private static float StatueX => BossRegistry.Current.StatueX;` in its place (the measured-value warning comment moves to `BossSpec.StatueX` — Task 8 already carries it). Change the branch condition at line 987 from `if (scene == "GG_Hornet_1")` to `if (scene == BossRegistry.Current.Scene)` — the dead-retry-pulse branch is boss-generic (a Godhome retry prompt is confirmed the same way in every arena). Update the two comments that name `GG_Hornet_1` in that region to say "the boss arena scene".

- [ ] **Step 2: `StateReader.cs`**

In `ReadBoss()`:

```csharp
                bossGo = GameObject.Find(BossRegistry.Current.ObjectName);
```

and gate the needle scan on the spec (a boss with no tracked projectile pays no per-call `GameObject.Find` and always reads inactive):

```csharp
            string needleName = BossRegistry.Current.NeedleName;
            if (needleName != null && needleGo == null)
                needleGo = GameObject.Find(needleName);
```

In `ConfirmAttunedChallenge()`, invoke with the spec's tier index:

```csharp
                method.Invoke(ui, new object[] { BossRegistry.Current.TierIndex });
```

One caveat to preserve in a comment on `ReadBoss`'s find: the boss cache is cleared on scene change (`OnSceneChange`), and `BossRegistry.Current` only changes at reset acceptance, which always precedes the arena (re)entry that repopulates the cache — so a boss switch can never leave a stale `bossGo` from the previous boss.

- [ ] **Step 3: Build**

Run: `mod/build.sh`
Expected: build succeeds.

- [ ] **Step 4: Grep for leftovers**

Run: `grep -rn "GG_Hornet_1\|Hornet Boss 1\|62.21\|MaxAttunedHornetHp" mod/*.cs`
Expected: hits only inside `BossRegistry.cs`'s hornet1 entry and inside comments describing Hornet-specific history (the `TickReset` doc comments explaining the win/death/truncation reset origins may keep naming Hornet as the worked example). No live code references.

- [ ] **Step 5: Commit**

```bash
git add mod/EpisodeManager.cs mod/StateReader.cs
git commit -m "Route every boss-specific fact in the mod through the current registry entry: arena scene, boss GameObject name, statue X, HP ceiling, tier index, and the optional tracked projectile. The reset macro's retry and statue branches are now boss-generic."
```

---

### Task 10: Full-suite green + docs sweep

**Files:**
- Modify: `docs/DOCS.md` (protocol section), `trainer/hkrl/env.py` (module docstring)

**Interfaces:** none new — a verification and documentation checkpoint.

- [ ] **Step 1: Run everything**

Run (from `trainer/`): `.venv/bin/python -m pytest`
Expected: all PASS. Fix anything that fails before proceeding (the usual suspects: a test constructing FakeGame with positional args, or asserting on reset message shape).

- [ ] **Step 2: Update docs**

1. `trainer/hkrl/env.py` line 1: docstring says `(v1)` → `(v2)`.
2. `docs/DOCS.md`: in the protocol section, document v2 — the reset request's required `boss` field, the `error` reply, the version gate; in the section describing observations, note the one-hot is per-boss (sized by the registry's FSM list) and that checkpoints are therefore boss-specific.

- [ ] **Step 3: Commit**

```bash
git add docs/DOCS.md trainer/hkrl/env.py
git commit -m "Document protocol v2 and per-boss observation spaces in DOCS.md."
```

---

### Task 11: In-game discovery session for Gruz Mother (user drives)

**Files:**
- Modify: `mod/DISCOVERED.md` (new sections; follow its "How to gather these values" conventions)

This task needs the real game and the user at the keyboard. Produce a new DISCOVERED.md section per item, each with the measured value and how it was measured. **A temporary `--boss gruz_mother` cannot run yet** (the registries have no entry), so measurements come from a normal human-played session with the mod's F1 overlay and ModLog, standing at / fighting the Gruz Mother statue in the Hall of Gods:

- [ ] **Step 1: Statue-stand X** — stand at the Gruz Mother statue in `GG_Workshop`, read Knight X off the F1 overlay. Record like section 3 did for Hornet (62.21).
- [ ] **Step 2: Arena scene name and bounds** — enter the Attuned fight; ModLog's scene-change line names the arena scene (expected `GG_Gruz_Mother` — record what's actually logged). Walk to the left wall, right wall, and note floor Y and usable ceiling off the F1 overlay, then derive center/half-width/height the way section 2 does.
- [ ] **Step 3: Boss GameObject name** — needed for `ObjectName`. Add a temporary ModLog line (or use the FSMLogger's object naming) that logs the root GameObject name of the scene's `HealthManager` owner during the fight; record the exact string.
- [ ] **Step 4: FSM state names** — with the FSMLogger active, play several full fights (win at least one, lose at least one) and collect the distinct state names of the boss's `Control` FSM. Record the list; it plus `UNKNOWN` becomes `fsm_states`.
- [ ] **Step 5: Attuned max HP + ceiling** — record `bossMaxHp` from the fight logs across the fights; pick a ceiling safely above the stable Attuned value and below the next tier's (fight one Ascended round to read its value, mirroring how Hornet's 1000 was chosen between 900 and 1186/1250).
- [ ] **Step 6: Tier gate + win detection sanity** — confirm the statue's challenge menu has the same three-tier layout (Attuned = tier index 0), and note anything odd about the death sequence (Gruz bursts into baby gruzzers; the `On.HealthManager.Die` hook should fire on the fatal blow regardless — this is fully verified in Task 13's smoke).
- [ ] **Step 7: Commit**

```bash
git add -f mod/DISCOVERED.md
git commit -m "Record the Gruz Mother discovery session in DISCOVERED.md: statue X, arena scene and bounds, boss GameObject name, FSM state list, and Attuned HP ceiling."
```

(Note: `docs/` is gitignored but `mod/DISCOVERED.md` is tracked normally — no `-f` actually needed; keep it, it is harmless.)

---

### Task 12: Transcribe Gruz Mother into both registries

**Files:**
- Modify: `trainer/hkrl/bosses.py`, `mod/BossRegistry.cs`
- Test: `trainer/tests/test_bosses.py`

**Interfaces:**
- Consumes: the measured values from Task 11's DISCOVERED.md sections. Every number below written as `<measured>` is copied verbatim from those sections — do not estimate any of them.
- Produces: `BOSSES["gruz_mother"]` and `BossRegistry.All["gruz_mother"]`.

- [ ] **Step 1: Write the failing tests**

Append to `trainer/tests/test_bosses.py`:

```python
def test_gruz_mother_is_registered_with_its_own_obs_space():
    spec = get_boss("gruz_mother")
    assert spec.id == "gruz_mother"
    assert spec.fsm_states[-1] == "UNKNOWN"
    # Different state list -> different obs size -> boss-specific policies.
    assert spec.fsm_states != get_boss("hornet1").fsm_states
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bosses.py -v -k gruz`
Expected: FAIL with `ValueError: unknown boss 'gruz_mother'`

- [ ] **Step 3: Add the entries**

`trainer/hkrl/bosses.py` — add to `BOSSES`, citing the DISCOVERED.md sections by name in the comment:

```python
    # Gruz Mother (Hall of Gods, Attuned). States and arena measured
    # <date>, DISCOVERED.md sections <n> and <n+1>.
    "gruz_mother": BossSpec(
        id="gruz_mother",
        fsm_states=(
            # <measured state names from the discovery session>,
            "UNKNOWN",
        ),
        arena_center_x=<measured>,
        arena_half_w=<measured>,
        floor_y=<measured>,
        arena_height=<measured>,
    ),
```

`mod/BossRegistry.cs` — add to `All`:

```csharp
                ["gruz_mother"] = new BossSpec
                {
                    Id = "gruz_mother",
                    Scene = "<measured scene name>",
                    ObjectName = "<measured GameObject name>",
                    StatueX = <measured>f,
                    MaxAttunedHp = <chosen ceiling>,
                    TierIndex = 0,
                    NeedleName = null,   // no tracked projectile
                },
```

- [ ] **Step 4: Run tests and build**

Run: `.venv/bin/python -m pytest tests/test_bosses.py tests/test_env.py -v` and `mod/build.sh`
Expected: all PASS; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add trainer/hkrl/bosses.py mod/BossRegistry.cs trainer/tests/test_bosses.py
git commit -m "Register Gruz Mother in both boss registries from the measured discovery values."
```

---

### Task 13: Real-game smoke against Gruz Mother (user present)

No file changes expected (fixes found here become their own commits). Rebuild + re-sign the mod (`run.sh` does this when stale) and run a short session:

- [ ] **Step 1: Episodes end-to-end** — `./.venv/bin/python scripts/train.py --boss gruz_mother --run-id gruz-smoke --timesteps 3000` (from `trainer/`, game managed by train.py as usual). Watch ModLog + the console for: the reset macro walking to the *Gruz* statue (correct StatueX), the fight going live in the measured scene with `bossMaxHp` under the ceiling, episodes producing nonzero `boss_damage_frac`.
- [ ] **Step 2: Win detection** — if the random-ish early policy can't win, verify the Die hook by winning manually once in the same setup (or replay once a checkpoint can win). A kill must log `won=True` and the burst-into-gruzzers death sequence must not produce a post-win loss.
- [ ] **Step 3: Loss + reset rhythm** — confirm deaths route through the retry prompt (dead-retry-pulse branch) and resets complete inside the budget.
- [ ] **Step 4: Resume guard live** — `--resume` the smoke run with no flag (works), then with `--boss hornet1` (must exit with the conflict message).
- [ ] **Step 5: Dashboard** — start a short run from the panel with the boss dropdown set to `gruz_mother`; confirm the run card shows the boss and the run's `config.jsonl` records it.
- [ ] **Step 6: Delete the smoke run(s)** via the dashboard's Delete (goes to trash), and commit any fixes made along the way.

---

### Task 14: Train Gruz Mother to winning (the "done" bar)

- [ ] **Step 1: Launch the real run** — from the dashboard or CLI: `--boss gruz_mother`, default instances/settings the Hornet runs use (N=2 async is proven), `--target-kl 0.05` recommended given the late-training findings. Budget generously (`--timesteps 500000`); Gruz should converge far sooner.
- [ ] **Step 2: Success criteria** — a sustained high win_rate plateau on the dashboard (the Hornet bar: ~0.85+; Gruz being far simpler, expect near-1.0). If learning stalls, characterize before tuning (the obs has no projectile entities; Gruz's threats are body-only, so the Hornet feature set should suffice).
- [ ] **Step 3: Wrap up** — record the outcome (win rate, steps to plateau) in the run's notes/memory, update README's boss list if guidance changed, and use the finishing-a-development-branch skill for merge.

---

## Self-Review (completed)

- **Spec coverage:** registry split (T1/T8), env per-boss obs (T2), protocol v2 + error + version gate (T3/T8), `--boss` + resume guard + config recording (T4), replay (T5), launcher whitelist + restart carry (T6), dashboard picker + display (T7), mod parameterization incl. needle/tier/statue (T9), docs (T10), discovery (T11), transcription (T12), smoke (T13), train-to-winning (T14). Error handling table in spec: unknown id (T3/T8), version mismatch (T3/T8), resume mismatch (T4), wrong tier (T9, per-boss ceiling).
- **Placeholder scan:** Task 12's `<measured>` markers are deliberate — they denote values that can only exist after Task 11's physical measurements, with the exact source named. No other TBDs.
- **Type consistency:** `boss` is always the string id at boundaries (env kwarg, reset message, argv, config.jsonl, launcher params); `BossSpec` objects never cross a process or protocol boundary. `resolve_boss` raises `ValueError` (converted to `sys.exit` in `main`), matching its tests; `run_boss` returns the id string.
