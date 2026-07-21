"""Launch a Hollow Knight instance with the RL bridge and wait for it.

The game binary is executed directly rather than through Steam so the launch
is a plain child process this script (and the supervisor) can signal and reap.
HKRL_PORT tells the mod which port to listen on.

Also the library the supervisor and the training script import: `launch`,
`shutdown` and `wait_for_port` are the three calls a `relaunch(slot)` needs.
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hkrl.cloneprep import prepare_clone_save  # noqa: E402

if sys.platform == "win32":
    DEFAULT_APP = Path(
        r"C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight"
        r"\hollow_knight.exe"
    )
else:
    DEFAULT_APP = Path(
        "~/Library/Application Support/Steam/steamapps/common/Hollow Knight/"
        "hollow_knight.app/Contents/MacOS/Hollow Knight"
    ).expanduser()

DEFAULT_PORT = 9020

# Unity's persistentDataPath on macOS is ~/Library/Application Support/
# <CFBundleIdentifier> -- for the stock game, "unity.Team Cherry.Hollow
# Knight". HOME redirection does NOT move it (disproven live 2026-07-20:
# a game launched with a redirected HOME still wrote the master ModLog),
# so isolation instead clones the whole .app per instance with a per-port
# bundle id, which moves that instance's save dir and ModLog wholesale.
# Windows resolves its equivalent (AppData/LocalLow/<company>/<product>)
# from values baked into the build, not the environment -- no isolation
# there yet.
MASTER_BUNDLE_ID = "unity.Team Cherry.Hollow Knight"
APP_SUPPORT = Path("~/Library/Application Support").expanduser()

SAVE_ISOLATION_SUPPORTED = sys.platform == "darwin"

# The master save directory per platform; None where the location is
# unknown (nothing to back up there).
if sys.platform == "darwin":
    MASTER_SAVE_DIR = APP_SUPPORT / MASTER_BUNDLE_ID
elif sys.platform == "win32":
    MASTER_SAVE_DIR = Path.home() / "AppData/LocalLow/Team Cherry/Hollow Knight"
else:
    MASTER_SAVE_DIR = None


def backup_saves(root: Path = None, keep: int = 10,
                 source: Path = None, stamp: str = None) -> Path | None:
    """Snapshot the master save directory; returns the snapshot path.

    Taken automatically before any run launches games. The game's own
    .bakNNN rotation is NOT sufficient protection: it lives inside the
    same directory, so anything that scrambles the directory itself -- or
    a corrupt save faithfully rotated into the backups -- takes the
    rotation with it. A concurrent-autosave accident destroyed the master
    slot live (2026-07-20) and was only recoverable because a manual
    snapshot habit existed; this makes that habit automatic.

    Snapshots land under <root>/save-backups/<timestamp>; only the newest
    `keep` are retained (a save dir is a few MB, so ten snapshots are
    noise). Returns None when no master save directory exists on this
    platform -- there is nothing to protect.
    """
    src = source if source is not None else MASTER_SAVE_DIR
    if src is None or not Path(src).exists():
        return None
    root = Path(root).expanduser() if root is not None \
        else Path("~/hkrl").expanduser()
    dest_root = root / "save-backups"
    stamp = stamp if stamp is not None else time.strftime("%Y%m%d_%H%M%S")
    dest, n = dest_root / stamp, 1
    while dest.exists():  # same-second runs must not overwrite each other
        n += 1
        dest = dest_root / f"{stamp}_{n}"
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    for old in sorted(d for d in dest_root.iterdir() if d.is_dir())[:-keep]:
        shutil.rmtree(old)
    return dest


def seed_save_dir(bundle_id: str, source: Path = None,
                  app_support: Path = None) -> Path:
    """Refresh <app_support>/<bundle_id> from the master save directory.

    Two game instances sharing one save directory autosave the same slot
    concurrently throughout a run (every bench/statue interaction), which
    corrupted the master save live (2026-07-20: both games save-on-exit in
    the same second; the next boot read slot 1 as empty and the boot macro
    started a new game -- the game's own .bakNNN rotation recovered it).

    Always refreshed from the master: whatever save churn a run produces in
    the clone is disposable, and every run starts from the parked-at-bench
    state the master holds.
    """
    # Module attribute resolved at call time, not bound as a default, so
    # tests can point APP_SUPPORT at a sandbox.
    if app_support is None:
        app_support = APP_SUPPORT
    src = source if source is not None else app_support / MASTER_BUNDLE_ID
    dst = app_support / bundle_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def seed_prefs(bundle_id: str, slot: int = 0) -> None:
    """Copy the master's PlayerPrefs domain onto a clone's, forced windowed.

    Unity keeps PlayerPrefs in ~/Library/Preferences/<bundle id>.plist --
    OUTSIDE persistentDataPath, so seeding the save directory alone leaves
    a clone booting with first-run defaults, whose extra menu prompts
    desync the boot macro's fixed input sequence (observed live 2026-07-20:
    both instances wandered into the settings menu). Routed through
    `defaults` rather than copying the plist file so cfprefsd's cache
    never serves a stale/negative entry for the clone's domain.

    The Screenmanager keys are then overridden to WINDOWED at a per-slot
    size: the master plays fullscreen at native res, and N fullscreen
    clones stack on one display -- the hidden one gets App Nap'd at ~0%
    CPU and drags the whole lockstep down to its crawl (observed live:
    "really slow", one game at 0.5% CPU, its macro wedging mid-menu).
    Strictly decreasing sizes mean a later (frontmost) window can never
    fully occlude an earlier one, so no clone is ever occlusion-suspended
    even before anyone tiles the windows properly.
    """
    exported = subprocess.run(["defaults", "export", MASTER_BUNDLE_ID, "-"],
                              check=True, capture_output=True).stdout
    subprocess.run(["defaults", "delete", bundle_id],
                   capture_output=True)  # fresh domain; ok if absent
    subprocess.run(["defaults", "import", bundle_id, "-"],
                   input=exported, check=True)
    width = max(640, 1280 - 160 * slot)
    height = max(360, 720 - 90 * slot)
    for key, value in [
        ("Screenmanager Fullscreen mode", 3),      # Unity FullScreenMode.Windowed
        ("Screenmanager Is Fullscreen mode", 0),
        ("Screenmanager Resolution Use Native", 0),
        ("Screenmanager Resolution Width", width),
        ("Screenmanager Resolution Height", height),
    ]:
        subprocess.run(["defaults", "write", bundle_id, key,
                        "-int", str(value)], check=True)


def prepare_instance(port: int, app: Path = None,
                     root: Path = None, sign: bool = True,
                     prefs: bool = True, slot: int = 0) -> Path:
    """Build this port's isolated game copy; returns its binary to launch.

    An APFS copy-on-write clone (cp -c: instant, near-zero disk) of the
    whole .app, its CFBundleIdentifier suffixed with the port so Unity
    derives a per-instance persistentDataPath -- own saves, own ModLog --
    then ad-hoc re-signed (the plist edit invalidates the signature, and
    unsigned launches die instantly with exit code 138). The instance's
    save dir is seeded from the master save on every call.

    The clone is rebuilt from the master app every time, so a mod rebuild
    (build.sh + codesign on the master) propagates on the next start.
    """
    app = Path(app) if app is not None else DEFAULT_APP
    bundle = app.parents[2]  # .../hollow_knight.app/Contents/MacOS/<bin>
    root = Path(root).expanduser() if root is not None \
        else Path("~/hkrl/instances").expanduser()
    clone = root / f"port-{port}" / bundle.name
    clone.parent.mkdir(parents=True, exist_ok=True)
    if clone.exists():
        shutil.rmtree(clone)
    subprocess.run(["cp", "-Rc", str(bundle), str(clone)], check=True)
    bundle_id = f"{MASTER_BUNDLE_ID}.hkrl{port}"
    subprocess.run(
        ["/usr/libexec/PlistBuddy", "-c",
         f"Set :CFBundleIdentifier {bundle_id}",
         str(clone / "Contents" / "Info.plist")],
        check=True)
    if sign:
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(clone)],
            check=True, capture_output=True)
    save_dir = seed_save_dir(bundle_id)
    prepare_clone_save(save_dir)
    if prefs:
        seed_prefs(bundle_id, slot=slot)
    return clone / "Contents" / "MacOS" / app.name


def wait_for_port(
    port: int,
    timeout: float = 120.0,
    host: str = "127.0.0.1",
    proc: subprocess.Popen | None = None,
) -> None:
    """Block until the port accepts a connection.

    The bridge only starts listening once the mod has loaded, which is well
    after the process itself exists -- so process liveness is not a usable
    *readiness* signal. It is a usable *failure* signal: a process that has
    already exited will never listen, so with `proc` supplied this raises at
    once naming the exit code instead of burning the full timeout. That
    matters because a game launched with stdout/stderr on DEVNULL reports a
    bad --app path or a mod that fails to load in no other way.

    `proc` is optional so callers holding only a port keep working.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            # Checked after the connect attempt, not before: a process that
            # exited *after* handing off a working bridge is not the case
            # being caught here, and connecting first keeps that ordering
            # from mattering.
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"process for port {port} exited with code "
                    f"{proc.returncode} before its bridge started listening"
                )
            time.sleep(0.5)
    raise TimeoutError(f"port {port} never accepted a connection within {timeout}s")


def launch(port: int, app: Path, visible: bool) -> subprocess.Popen:
    # SteamAppId/SteamGameId are the launch context Steam exports to its
    # children; the Steam build's DRM check looks for them in the
    # environment. Executed directly without them, the game asks Steam to
    # relaunch it (steam://run/367520 in Steam's console log) and quits
    # ~15s after boot -- at the title menu, before any bridge traffic.
    # steam_appid.txt beside the binary or in the cwd does NOT satisfy the
    # check on macOS; only the env vars do (verified live, 2026-07-19).
    # Unverified on Windows, where steam_appid.txt reportedly also works;
    # the env vars are kept as the single cross-platform mechanism.
    env = dict(os.environ, HKRL_PORT=str(port),
               SteamAppId="367520", SteamGameId="367520")
    # Detach the game from the terminal's Ctrl-C: Ctrl-C at the trainer must
    # interrupt the trainer alone, not kill the game out from under the
    # supervisor and the final checkpoint save. shutdown() is the one
    # intended kill path. On POSIX that means a new session (out of the
    # terminal's foreground process group); on Windows, a new process group
    # (out of the console's CTRL_C_EVENT broadcast).
    if sys.platform == "win32":
        detach = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        detach = {"start_new_session": True}
    return subprocess.Popen(
        [str(app)],
        env=env,
        stdout=subprocess.DEVNULL if not visible else None,
        stderr=subprocess.DEVNULL if not visible else None,
        **detach,
    )


def shutdown(procs: list, grace: float = 15.0) -> None:
    """Terminate every process and block until each has actually exited.

    A game process runs its own Unity/Mono save-on-exit routine after
    SIGTERM, which can take several seconds; 15s is generous enough to let
    that finish under normal conditions. Any process still alive once the
    shared deadline passes is SIGKILLed and reaped, so a hung or
    signal-ignoring process can never outlive this call and be left holding
    the bridge port for a subsequent launch to collide with.

    On Windows, terminate() IS TerminateProcess -- an immediate hard kill
    with no save-on-exit window. That is acceptable here: the training save
    is parked at the Hall of Gods bench and the mod never writes game saves,
    so there is nothing in memory worth flushing. The escalation path is
    then a no-op (kill() == terminate()), but the wait-and-reap contract
    holds identically.
    """
    for p in procs:
        p.terminate()
    deadline = time.monotonic() + grace
    for p in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", type=Path, default=DEFAULT_APP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="first bridge port; instance i listens on port+i")
    ap.add_argument("--instances", type=int, default=1)
    args = ap.parse_args()

    # Fail fast on squatted ports BEFORE launching anything: the mod's
    # TcpListener.Start() throws on an in-use port and the game then runs
    # bridgeless, while wait_for_port happily greets the squatter -- observed
    # live (2026-07-20) when a second instance's port 9021 turned out to be
    # the dashboard's old default.
    for i in range(args.instances):
        port = args.port + i
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                pass
        except OSError:
            continue
        raise SystemExit(
            f"port {port} is already accepting connections -- another "
            f"process (a dashboard? a leftover game?) holds it. Free it or "
            f"pick a different --port range."
        )

    backup = backup_saves()
    if backup is not None:
        print(f"master save backed up to {backup}", flush=True)

    # Same save-isolation rule as train.py: instances sharing one save slot
    # autosave over each other. Manual multi-instance gates get the same
    # per-port app clones a training fleet would.
    apps = [args.app] * args.instances
    if SAVE_ISOLATION_SUPPORTED:
        apps = [prepare_instance(args.port + i, args.app, slot=i)
                for i in range(args.instances)]
    else:
        print("WARNING: save isolation is not implemented on this "
              "platform; all instances share one save slot and "
              "concurrent autosaves can corrupt it.", flush=True)

    procs = []
    try:
        # Launch all, then wait all, so the games boot in parallel (a cold
        # boot is tens of seconds each) -- same shape as GameFleet.start().
        for i in range(args.instances):
            procs.append(launch(args.port + i, apps[i], visible=True))
            print(f"launched: port={args.port + i}", flush=True)
        for i, proc in enumerate(procs):
            wait_for_port(args.port + i, proc=proc)
            print(f"bridge ready on {args.port + i}.", flush=True)
        print("Ctrl-C to stop.", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(procs)


if __name__ == "__main__":
    main()
