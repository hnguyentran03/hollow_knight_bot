"""Filesystem and port layout for parallel game instances.

Each instance runs with its own HOME so Unity resolves
Application.persistentDataPath to a private directory. Without this every
instance shares one save directory and one ModLog.txt.
"""
import shutil
from pathlib import Path

UNITY_SAVE_SUBPATH = "Library/Application Support/unity.Team Cherry.Hollow Knight"


def instance_home(n: int, root: Path) -> Path:
    return Path(root) / "instances" / str(n)


def port_for(n: int, base_port: int = 9020) -> int:
    return base_port + n


def provision(n: int, root: Path, seed_from: Path) -> Path:
    """Create instance n's HOME with a seeded save directory.

    Idempotent: an already-provisioned instance keeps its existing save, so
    relaunching between training runs does not reset game progress. A save
    counts as provisioned only when every seed file is present; if it's not
    (an earlier copy never finished), the directory is rebuilt from a clean
    seed instead of being treated as ready.
    """
    home = instance_home(n, root)
    saves = home / UNITY_SAVE_SUBPATH
    seed_files = [item for item in Path(seed_from).iterdir() if item.is_file()]
    fully_seeded = saves.exists() and all(
        (saves / item.name).exists() for item in seed_files
    )
    if not fully_seeded:
        # Copy into a temp dir next to the final path (same filesystem, so
        # the rename below is atomic) and only expose it at `saves` once
        # every file has landed. A process killed mid-copy leaves the
        # partial state in the temp dir, never at `saves`, so the next call
        # sees an incomplete (or absent) save and redoes the copy from a
        # clean temp dir rather than mistaking a partial copy for a
        # finished one.
        tmp = saves.with_name(saves.name + ".provision-tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            for item in seed_files:
                shutil.copy2(item, tmp / item.name)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        shutil.rmtree(saves, ignore_errors=True)
        tmp.rename(saves)
    return home
