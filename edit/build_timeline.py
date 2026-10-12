#!/usr/bin/env python3
"""Expand a cut list into a render.py timeline for still-frame edits.

A cut list (edit/cuts/*.json) names, per shot: where it starts on the beat
grid, which frame (framing.json key, e.g. "abudhabi21_039"), a camera move,
and cut effects. This script resolves frames to files, turns moves into
zoom/pan keyframes on top of the frame's framing (crop centre / zoom), checks
every sampled window of the move against the burned-in keep-out zones, turns
cut effects into timed fx, and writes the timeline.

Usage: build_timeline.py edit/cuts/v1.json edit/timeline_v1.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_framing import MARGIN, hits, key, window  # noqa: E402

FPS = 30

# camera moves for stills: zoom multiplier keyframes, horizontal drift (fraction of width)
MOVES = {
    "hold": dict(zoom=[[0, 1.02], [1, 1.04]], pan=0.0),
    "push": dict(zoom=[[0, 1.0], [1, 1.08]], pan=0.0),
    "pushfast": dict(zoom=[[0, 1.0], [1, 1.16]], pan=0.0),
    "slowpush": dict(zoom=[[0, 1.0], [1, 1.12]], pan=0.0),
    "pull": dict(zoom=[[0, 1.1], [1, 1.0]], pan=0.0),
    "panl": dict(zoom=[[0, 1.06], [1, 1.06]], pan=-0.05),
    "panr": dict(zoom=[[0, 1.06], [1, 1.06]], pan=0.05),
}


class Grid:
    def __init__(self, path):
        b = json.load(open(path))
        self.beats = np.array(b["beats"])
        self.per = float(np.median(np.diff(self.beats)))

    def t(self, v):
        if isinstance(v, (int, float)):
            return float(v)
        return float(self.beats[0] + float(v[1:]) * self.per)  # constant-tempo grid


def fx_for(kind, amt, t0, t1, per):
    """Cut effect at a cut placed at t0 (shot runs t0..t1)."""
    f = 1.0 / FPS
    if kind == "dip":      # signature luma crash across the cut, highlights survive
        return [dict(type="dip", t=t0 - 2 * f, dur=4 * f, amt=amt or 0.9, env="tri")]
    if kind == "flash":    # only for the biggest hits
        return [dict(type="flash", t=t0, dur=0.16, amt=amt or 0.85, env="decay"),
                dict(type="bloom", t=t0, dur=min(0.6, t1 - t0), amt=0.6, env="decay")]
    if kind == "bloom":
        return [dict(type="bloom", t=t0, dur=min(0.9, t1 - t0), amt=amt or 0.6, env="decay")]
    if kind == "punch":
        return [dict(type="punch", t=t0, dur=min(0.3, t1 - t0), amt=amt or 0.6, env="decay")]
    if kind == "shake":    # intensity follows the hit
        return [dict(type="shake", t=t0, dur=t1 - t0, amt=amt or 0.6, freq=1.6, env="decay")]
    if kind == "whip":
        return [dict(type="whip", t=t0 - 0.1, dur=0.2, amt=amt or 1.0, angle=0)]
    if kind == "zoomtrans":
        return [dict(type="zoomtrans", t=t0 - 0.12, dur=0.24, amt=amt or 1.0)]
    if kind == "rgb":
        return [dict(type="rgb", t=t0, dur=min(0.25, t1 - t0), amt=amt or 1.0, jitter=True)]
    if kind == "glitch":   # covers the whole shot
        return [dict(type="glitch", t=t0, dur=t1 - t0, amt=amt or 0.6, env="hold"),
                dict(type="rgb", t=t0, dur=t1 - t0, amt=0.8, jitter=True, env="hold")]
    if kind == "strobe":
        return [dict(type="strobe", t=t0, dur=t1 - t0, amt=amt or 0.8, period=1, env="hold")]
    if kind == "fadein":
        return [dict(type="exposure", t=t0, dur=amt or 1.0, amt=-4.0, env="out")]
    if kind == "fadeout":
        d = amt or 0.5
        return [dict(type="exposure", t=t1 - d, dur=d, amt=-5.0, env="in")]
    raise SystemExit(f"unknown cut effect {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cuts")
    ap.add_argument("out")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(a.cuts))
    spec = json.load(open(a.cuts))
    rel = lambda p: os.path.normpath(os.path.join(here, p))  # noqa: E731
    out_dir = os.path.dirname(os.path.abspath(a.out))
    to_out = lambda p: os.path.relpath(p, out_dir)  # noqa: E731
    framing = json.load(open(rel(spec["framing"])))
    grid = Grid(rel(spec["beats"]))
    files = {}
    for d in spec["frames"]:  # first folder that has the frame wins (upscaled before originals)
        for p in glob.glob(os.path.join(rel(d), "*")):
            files.setdefault(key(os.path.basename(p)), p)
    end = grid.t(spec["end"])
    rows = spec["shots"]
    shots, fx, used, problems = [], [], {}, []
    for k, row in enumerate(rows):
        at, name, move = row[0], row[1], row[2]
        opt = row[3] if len(row) > 3 else {}
        # snap cuts to the nearest frame (the renderer starts a shot on the first
        # frame at/after t0, which would otherwise land cuts up to a frame late)
        snap = lambda x: round(x * FPS) / FPS  # noqa: E731
        t0 = 0.0 if k == 0 else snap(grid.t(at))
        t1 = snap(grid.t(rows[k + 1][0])) if k + 1 < len(rows) else end
        if t1 - t0 < 2.0 / FPS:
            problems.append(f"shot {k} {name} shorter than 2 frames")
        fr = framing["frames"].get(name)
        if fr is None or name not in files:
            raise SystemExit(f"unknown frame {name}")
        if fr.get("skip"):
            problems.append(f"shot {k} uses skipped frame {name}: {fr['skip']}")
        used[name] = used.get(name, 0) + 1
        mv = MOVES[move]
        base_z, fy = fr.get("zoom", 1.0), fr.get("fy", 0.5)
        zoom = [[u, round(base_z * z, 4)] for u, z in mv["zoom"]]
        fx0 = fr["fx"]
        pan = opt.get("pan", mv["pan"])
        # keep the pan inside the clear range: shrink it until every sampled window passes
        boxes = framing["keepout_all"] + fr.get("keepout", [])
        sw, sh = 640, 360
        for shrink in (1.0, 0.6, 0.3, 0.0):
            p = pan * shrink
            focus = [[0, round(fx0 - p / 2, 4)], [1, round(fx0 + p / 2, 4)]] if p else fx0
            ok = True
            for u in np.linspace(0, 1, 9):
                z = float(np.interp(u, [q[0] for q in zoom], [q[1] for q in zoom]))
                fxu = fx0 + p * (u - 0.5)
                if any(hits(window(sw, sh, fxu, z, fy), b) for b in boxes):
                    ok = False
                    break
            if ok:
                break
        if not ok:
            problems.append(f"shot {k} {name}: window hits keep-out even without pan")
        s = dict(src=to_out(files[name]), t0=round(t0, 4), t1=round(t1, 4), focus=focus,
                 fy=fy, zoom=zoom, grade=opt.get("grade", spec.get("grade", "base")))
        if "exposure" in opt:
            s["exposure"] = opt["exposure"]
        shots.append(s)
        for item in opt.get("fx", []):
            kind, _, amt = item.partition(":")
            fx += fx_for(kind, float(amt) if amt else None, t0, t1, grid.per)
    for f in spec.get("fx", []):  # free-form fx in beat refs
        f = dict(f)
        f["t"] = round(grid.t(f["t"]), 4)
        if "t_end" in f:
            f["t_end"] = round(grid.t(f["t_end"]), 4)
        fx.append(f)
    for f in fx:
        f["t"] = round(max(0.0, f["t"]), 4)
        if "dur" in f:
            f["dur"] = round(f["dur"], 4)
    tl = dict(fps=FPS, width=1080, height=1920, music=to_out(rel(spec["music"])),
              music_start=spec.get("music_start", 0.0), duration=round(end, 4),
              fade_out=spec.get("fade_out", 0.3), grain=spec.get("grain", 0.04),
              loudness=spec.get("loudness", -10.0), true_peak=spec.get("true_peak", -1.0),
              crf=spec.get("crf", 15), maxrate=spec.get("maxrate", "40M"),
              bufsize=spec.get("bufsize", "80M"), shots=shots, fx=sorted(fx, key=lambda f: f["t"]))
    with open(a.out, "w") as fh:
        json.dump(tl, fh, indent=1)
    lens = np.diff([s["t0"] for s in shots] + [end])
    dup = {n: c for n, c in used.items() if c > 1}
    print(f"{os.path.basename(a.out)}: {len(shots)} shots, {len(used)} unique frames, "
          f"{end:.2f}s, shot length {lens.min():.2f}-{lens.max():.2f}s (median {np.median(lens):.2f})")
    if dup:
        print("  repeated frames:", dup)
    for p in problems:
        print("  PROBLEM:", p)
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
