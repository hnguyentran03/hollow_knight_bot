#!/usr/bin/env python3
"""Serve the training dashboard: live status and learning curves for every
run under <root>/runs, read straight from the run directories -- plus a
launch panel that can start, resume, and stop training runs (the one
mutating surface; everything else only reads, and runs it launches
survive this server exiting).
"""
import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hkrl.dashboard import make_server  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("~/hkrl").expanduser())
    ap.add_argument("--port", type=int, default=9700,
                    help="HTTP port (default 9700 -- far from the game "
                         "bridges, which count up from 9020 one port per "
                         "instance; the old default 9021 collided with the "
                         "second instance's bridge)")
    ap.add_argument("--open", action="store_true",
                    help="open the dashboard in the default browser")
    args = ap.parse_args()

    server = make_server(root=args.root, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"dashboard on {url} (root {args.root})", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
