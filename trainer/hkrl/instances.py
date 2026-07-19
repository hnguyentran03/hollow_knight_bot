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
    relaunching between training runs does not reset game progress.
    """
    home = instance_home(n, root)
    saves = home / UNITY_SAVE_SUBPATH
    if not saves.exists():
        saves.mkdir(parents=True)
        for item in Path(seed_from).iterdir():
            if item.is_file():
                shutil.copy2(item, saves / item.name)
    return home
