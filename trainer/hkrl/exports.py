"""Export a generation to a durable, run-independent location.

An export is the checkpoint pair under fixed names plus a manifest, in
<root>/exports/<name>/ -- one canonical place the play daemon
(scripts/play.py) and the in-game bot menu (mod/ExportsCatalog.cs) both
read, surviving run-dir cleanup. The manifest's boss fields are load-
bearing: exports are boss-specific by construction (the FSM state list
sizes the observation one-hot, see hkrl/bosses.py), so consumers read the
boss from here and never guess it from the name.
"""
import json
import shutil
import time
from pathlib import Path

from hkrl.bosses import BOSSES
from hkrl.generations import (MANIFEST_NAME, checkpoint_paths,
                              latest_checkpoint)
from hkrl.rundata import read_jsonl, run_boss

EXPORTS_DIR = "exports"
MODEL_NAME = "model.zip"
VECNORM_NAME = "vecnorm.pkl"
EXPORT_MANIFEST = "manifest.json"


def export_generation(root, run_dir, gen=None, name=None,
                      force=False) -> Path:
    """Copy one generation's checkpoint pair into <root>/exports/<name>/;
    returns the export directory.

    Default gen is the run's latest complete checkpoint; default name is
    <run_id>_gen<NNNN>. An existing export of the same name is refused
    (ValueError) unless force=True.
    """
    run_dir = Path(run_dir).expanduser()
    if gen is None:
        gen, weights, vecnorm = latest_checkpoint(run_dir)  # FileNotFoundError when none
    else:
        weights, vecnorm = checkpoint_paths(run_dir, gen)
        if not (weights.exists() and vecnorm.exists()):
            raise ValueError(
                f"generation {gen} of {run_dir.name!r} has no complete "
                f"checkpoint")
    boss = run_boss(run_dir)
    boss_display = BOSSES[boss].display_name if boss in BOSSES else boss
    name = name or f"{run_dir.name}_gen{gen:04d}"
    dest = Path(root).expanduser() / EXPORTS_DIR / name
    if dest.exists():
        if not force:
            raise ValueError(
                f"export {name!r} already exists; pass force to overwrite")
        # An export is a copy, recreatable from its run at any time, so
        # unlike run dirs (launcher's trash-not-rmtree rule) replacing one
        # outright is safe -- and leaves no stale files behind.
        shutil.rmtree(dest)
    stats = next((line for line in read_jsonl(run_dir / MANIFEST_NAME)
                  if line.get("gen") == gen), None)
    dest.mkdir(parents=True)
    shutil.copy2(weights, dest / MODEL_NAME)
    shutil.copy2(vecnorm, dest / VECNORM_NAME)
    manifest = {
        "name": name,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "gen": gen,
        "timestep": (stats or {}).get("timestep"),
        "boss": boss,
        "boss_display": boss_display,
        "stats": stats,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (dest / EXPORT_MANIFEST).write_text(json.dumps(manifest, indent=2))
    return dest


def exported_generations(root, run_id) -> set:
    """Generation numbers of run_id that already have an export under
    <root>/exports, read from the export manifests (the directory name is
    user-chosen, so only the manifest knows the source run). Unreadable
    entries are skipped: this feeds a display flag, never a guard."""
    exports_dir = Path(root).expanduser() / EXPORTS_DIR
    gens = set()
    try:
        dirs = list(exports_dir.iterdir())
    except OSError:
        return gens
    for d in dirs:
        try:
            manifest = json.loads((d / EXPORT_MANIFEST).read_text())
        except (OSError, ValueError):
            continue
        if manifest.get("run_id") == run_id \
                and isinstance(manifest.get("gen"), int):
            gens.add(manifest["gen"])
    return gens
