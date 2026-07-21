"""Pre-configure a disposable game clone's save so the boot macro's blind
profile-select confirm lands on the Godhome save: only that profile is left in
the clone's save dir, so there is nothing else to mis-select. Operates ONLY on
a clone's seeded save directory -- never the master. (Difficulty is handled at
runtime in the mod, not here.)
"""
import re
from pathlib import Path

GODHOME_SLOT = 1

_PROFILE_RE = re.compile(r"user(\d+)")
# A per-port clone's save dir ends in ".hkrl<port>"; the master is the same
# name without that suffix. This module deletes save files, so it must only
# ever run against a clone.
_CLONE_DIR_RE = re.compile(r"\.hkrl\d+$")


def _strip_foreign_profiles(save_dir, godhome_slot: int = GODHOME_SLOT) -> list:
    """Delete every non-Godhome profile file; return the deleted paths.

    A profile file is any entry whose name starts with `user<N>` for an integer
    N (its .dat, version-tagged .dat, .modded.json, and .bak variants all
    match). Keeping only slot `godhome_slot` leaves the profile-select screen a
    single save, so a blind confirm cannot load the wrong one. Non-profile files
    (shared.dat, ModLog.txt, ...) do not match the leading `user<digit>`;
    directories are skipped (unlink on a dir raises).

    Module-private on purpose: this is the raw delete primitive, and the only
    route to it is prepare_clone_save's clone-name guard below -- exporting it
    unguarded would leave a one-import path to deleting master profiles.
    """
    removed = []
    for entry in Path(save_dir).iterdir():
        m = _PROFILE_RE.match(entry.name)
        if m and entry.is_file() and int(m.group(1)) != godhome_slot:
            entry.unlink()
            removed.append(entry)
    return removed


def prepare_clone_save(save_dir, godhome_slot: int = GODHOME_SLOT) -> None:
    """Prepare a freshly-seeded clone save dir for the blind profile-select.

    Refuses any dir that is not a per-port clone (name ending `.hkrl<port>`):
    this deletes save files, and pointing it at the master would destroy real
    profiles. This project has already lost save data once (see backup_saves);
    the guard makes that class of caller bug impossible, not merely unlikely.
    """
    save_dir = Path(save_dir)
    if not _CLONE_DIR_RE.search(save_dir.name):
        raise ValueError(
            f"refusing to prep {save_dir.name!r}: not a per-port clone save "
            "dir (expected a name ending in '.hkrl<port>'); the master save "
            "must never be modified")
    _strip_foreign_profiles(save_dir, godhome_slot)
