#!/usr/bin/env python3
"""Export a run's generation to <root>/exports/<name>/.

The export (model.zip + vecnorm.pkl + manifest.json) is what the in-game
bot menu lists and scripts/play.py plays; it survives run-dir cleanup.
Default: the run's latest complete generation, named <run_id>_gen<NNNN>.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkrl.exports import export_generation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--gen", type=int, default=None,
                    help="generation number (default: the run's latest)")
    ap.add_argument("--name", default=None,
                    help="export name (default: <run_id>_gen<NNNN>)")
    ap.add_argument("--root", type=Path, default=Path("~/hkrl").expanduser(),
                    help="hkrl root; the export lands under <root>/exports")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing export of the same name")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    try:
        dest = export_generation(args.root, args.run_dir, gen=args.gen,
                                 name=args.name, force=args.force)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    print(f"exported to {dest}")


if __name__ == "__main__":
    main()
