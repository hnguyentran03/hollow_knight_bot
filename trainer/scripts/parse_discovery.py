#!/usr/bin/env python3
"""Summarize a discovery session's DISCOVERY lines into registry-ready values.

The mod's DiscoveryLogger (F4 in-game) writes "DISCOVERY ..." lines to
ModLog.txt while a human plays a fight against a boss the registries don't
know yet. This script reduces a ModLog to what a new BossSpec /
BossRegistry entry needs: boss GameObject candidates ranked by peak HP,
each candidate's Control-FSM state vocabulary in first-seen order, arena
bounds per scene from the knight's extremes, and statue-stand X readings.

Usage:
    python scripts/parse_discovery.py path/to/ModLog.txt
"""
import argparse
import re
from collections import defaultdict

STATE_RE = re.compile(r"DISCOVERY state go='(?P<go>.*?)' state='(?P<state>.*?)'")
CANDIDATE_RE = re.compile(
    r"DISCOVERY candidate go='(?P<go>.*?)' hp=(?P<hp>\d+) scene=(?P<scene>\S*)")
ARENA_RE = re.compile(
    r"DISCOVERY arena scene=(?P<scene>\S*) kxRange=\[(?P<min>-?[\d.]+|NaN), "
    r"(?P<max>-?[\d.]+|NaN)\] floorY=(?P<floor>-?[\d.]+|NaN) maxKy=(?P<top>-?[\d.]+|NaN)")
STATUE_RE = re.compile(r"DISCOVERY statue knightX=(?P<x>-?[\d.]+) scene=(?P<scene>\S*)")


def summarize(lines):
    states = defaultdict(list)   # go -> distinct states, first-seen order
    candidates = {}              # go -> {"hp": peak, "scenes": set}
    arenas = {}                  # scene -> latest arena reading (floats)
    statue_xs = []
    for line in lines:
        m = STATE_RE.search(line)
        if m:
            if m["state"] not in states[m["go"]]:
                states[m["go"]].append(m["state"])
            continue
        m = CANDIDATE_RE.search(line)
        if m:
            c = candidates.setdefault(m["go"], {"hp": 0, "scenes": set()})
            c["hp"] = max(c["hp"], int(m["hp"]))
            c["scenes"].add(m["scene"])
            continue
        m = ARENA_RE.search(line)
        if m:
            arenas[m["scene"]] = {
                k: (float("nan") if m[k] == "NaN" else float(m[k]))
                for k in ("min", "max", "floor", "top")}
            continue
        m = STATUE_RE.search(line)
        if m:
            statue_xs.append(float(m["x"]))
    return {"states": dict(states), "candidates": candidates,
            "arenas": arenas, "statue_xs": statue_xs}


def report(s):
    out = ["boss candidates (by peak HP; the boss is almost always the top one):"]
    for go, c in sorted(s["candidates"].items(), key=lambda kv: -kv[1]["hp"]):
        out.append(f"  go='{go}' peak hp={c['hp']} scenes={sorted(c['scenes'])}")
    out.append("")
    out.append("Control-FSM states (first-seen order; append \"UNKNOWN\" when transcribing):")
    for go, names in s["states"].items():
        out.append(f"  go='{go}' ({len(names)} states):")
        out.extend(f'    "{n}",' for n in names)
    out.append("")
    out.append("arena per scene (knight extremes; re-tag both walls if the range looks short):")
    for scene, a in s["arenas"].items():
        out.append(
            f"  scene={scene}: center_x={(a['min'] + a['max']) / 2:.2f} "
            f"half_w={(a['max'] - a['min']) / 2:.2f} floor_y={a['floor']:.2f} "
            f"height={a['top'] - a['floor']:.2f} "
            f"(raw min={a['min']:.2f} max={a['max']:.2f} top={a['top']:.2f})")
    out.append("")
    xs = s["statue_xs"]
    out.append("statue knightX readings: "
               + (", ".join(f"{x:.2f}" for x in xs) if xs else "none")
               + "  (use the one from standing settled at the target statue)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("modlog", help="path to ModLog.txt from a discovery session")
    args = ap.parse_args()
    with open(args.modlog, errors="replace") as f:
        print(report(summarize(f)))


if __name__ == "__main__":
    main()
