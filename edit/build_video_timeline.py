#!/usr/bin/env python3
"""Expand a video cut list (edit/cuts_video/*.json) into a render.py timeline.

A cut list places source video moments on the song's constant beat grid:

  "grid":  {"phase": P, "period": T}   song beat k sits at P + k*T seconds
  "start": "b11", "end": "b39"        the slice of the song the edit runs over
  "sources": {"key": "path to clip"}
  "shots": [[at, source_key, in_seconds, {options}], ...]

`at` is a song beat ("b15", "b14.5") or seconds of edit time; a shot runs
until the next shot's `at`. Cuts are snapped to the nearest output frame.
Options are passed through to render.py (speed, reverse, stutter, mblur,
interp, focus, fy, zoom, grade, exposure) plus "fx": a list of cut-effect
macros ("dip", "flash:0.9", "punch:0.8", "white:0.3", ...) or full fx dicts
whose time is either "t"/"t_end" (beat refs) or "u0"/"u1" (fractions of the
shot). Free-form fx at the top level use beat refs.

Every shot is checked against the source: the consumed source range must fit
in the clip and should not cross a scene cut found by scan_motion.py
(media/work/scan/<clip>.json, if present), and the reframed window (zoom,
focus, fy) must stay clear of the burned-in graphics listed under "keepout"
(per source, plus per-shot "keepout" boxes; [x0, y0, x1, y1] fractions).

Usage: build_video_timeline.py edit/cuts_video/v1.json edit/timeline_v1_16x9.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


def fx_macro(kind, amt, t0, t1, fps):
    """Cut effect for a shot running t0..t1 (edit seconds)."""
    f = 1.0 / fps
    d = t1 - t0
    if kind == "dip":      # signature luma crash across the cut, highlights survive
        return [dict(type="dip", t=t0 - 2 * f, dur=4 * f, amt=amt or 0.9, env="tri")]
    if kind == "black":    # held drop into darkness at the start of the shot, amt = frames
        n = int(amt or 3)
        return [dict(type="dip", t=t0, dur=n * f, amt=1.0, env="hold")]
    if kind == "flash":    # only for the biggest hits
        return [dict(type="flash", t=t0, dur=0.16, amt=amt or 0.85, env="decay"),
                dict(type="bloom", t=t0, dur=min(0.6, d), amt=0.6, env="decay")]
    if kind == "white":    # blow out to white over the end of the shot, amt = seconds
        w = min(amt or 0.3, d)
        return [dict(type="exposure", t=t1 - w, dur=w, amt=2.6, env="in"),
                dict(type="bloom", t=t1 - w, dur=w, amt=1.0, env="in")]
    if kind == "bloom":
        return [dict(type="bloom", t=t0, dur=min(0.9, d), amt=amt or 0.6, env="decay")]
    if kind == "punch":
        return [dict(type="punch", t=t0, dur=min(0.3, d), amt=amt or 0.6, env="decay")]
    if kind == "shake":    # intensity follows the hit
        return [dict(type="shake", t=t0, dur=d, amt=amt or 0.6, freq=1.6, env="decay")]
    if kind == "whip":     # centred on the cut into this shot
        return [dict(type="whip", t=t0 - 0.1, dur=0.2, amt=amt or 1.0, angle=0)]
    if kind == "zoomtrans":
        return [dict(type="zoomtrans", t=t0 - 0.12, dur=0.24, amt=amt or 1.0)]
    if kind == "rgb":
        return [dict(type="rgb", t=t0, dur=min(0.25, d), amt=amt or 1.0, jitter=True)]
    if kind == "glitch":   # whole shot
        return [dict(type="glitch", t=t0, dur=d, amt=amt or 0.6, env="hold"),
                dict(type="rgb", t=t0, dur=d, amt=0.8, jitter=True, env="hold")]
    if kind == "glitchend":  # glitch shift over the last amt seconds
        w = min(amt or 0.15, d)
        return [dict(type="glitch", t=t1 - w, dur=w, amt=0.8, env="hold"),
                dict(type="rgb", t=t1 - w, dur=w, amt=1.0, jitter=True, env="hold")]
    if kind == "strobe":
        return [dict(type="strobe", t=t0, dur=d, amt=amt or 0.8, period=1, env="hold")]
    if kind == "invert":   # negative, whole shot
        return [dict(type="invert", t=t0, dur=d, amt=amt or 1.0, env="hold")]
    if kind == "fadein":
        return [dict(type="exposure", t=t0, dur=amt or 1.0, amt=-4.0, env="out")]
    if kind == "fadeout":
        w = amt or 0.5
        return [dict(type="exposure", t=t1 - w, dur=w, amt=-5.0, env="in")]
    raise SystemExit(f"unknown cut effect {kind}")


def probe(path):
    out = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)
    s = out["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return dict(w=s["width"], h=s["height"], fps=float(n) / float(d),
                dur=float(out["format"]["duration"]))


def consumed(speed, n, fps, reverse=False):
    """Source seconds used by a shot of n output frames (render.py's time remap)."""
    u = np.arange(n + 1) / max(1, n)
    if isinstance(speed, list):
        sp = np.interp(u, [p[0] for p in speed], [p[1] for p in speed])
    else:
        sp = np.full(n + 1, float(speed))
    off = np.concatenate([[0], np.cumsum((sp[1:] + sp[:-1]) / 2) / fps])
    return float(off.max() - off.min())


def kf(spec, u):
    if not isinstance(spec, list):
        return float(spec)
    return float(np.interp(u, [p[0] for p in spec], [p[1] for p in spec]))


def window(z, fx, fy):
    """Visible source window (fractions) for render.py's 16:9 -> 16:9 reframing."""
    hw = 0.5 / z
    cx = min(max(fx, hw), 1 - hw)
    cy = min(max(0.5 + (fy - 0.5) * (1 - 1 / z), hw), 1 - hw)
    return cx - hw, cy - hw, cx + hw, cy + hw


def hits(win, box, margin=0.004):
    return not (box[2] + margin <= win[0] or box[0] - margin >= win[2] or
                box[3] + margin <= win[1] or box[1] - margin >= win[3])


PASS = ("speed", "reverse", "stutter", "mblur", "interp", "focus", "fy", "zoom", "grade", "exposure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cuts")
    ap.add_argument("out")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(a.cuts))
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    spec = json.load(open(a.cuts, encoding="utf-8"))
    rel = lambda p: os.path.normpath(os.path.join(here, p))  # noqa: E731
    out_dir = os.path.dirname(os.path.abspath(a.out))
    to_out = lambda p: os.path.relpath(p, out_dir).replace("\\", "/")  # noqa: E731
    fps = spec.get("fps", 30)
    phase, period = spec["grid"]["phase"], spec["grid"]["period"]

    def song(v):
        return phase + float(v[1:]) * period if isinstance(v, str) else None

    t_start = song(spec["start"])

    def t(v):
        """Edit time of a beat ref ("b15" = song beat 15) or of plain seconds."""
        return song(v) - t_start if isinstance(v, str) else float(v)

    snap = lambda x: round(x * fps) / fps  # noqa: E731
    end = snap(t(spec["end"]))
    sources = {k: rel(p) for k, p in spec["sources"].items()}
    info = {k: probe(p) for k, p in sources.items()}
    scenes = {}
    for k, p in sources.items():
        sj = os.path.join(root, "media", "work", "scan",
                          os.path.splitext(os.path.basename(p))[0] + ".json")
        if os.path.exists(sj):
            scenes[k] = [(s["start"], s["end"]) for s in json.load(open(sj))["shots"]]
    rows = spec["shots"]
    shots, fx, problems, table = [], [], [], []
    for k, row in enumerate(rows):
        at, key, t_in = row[0], row[1], float(row[2])
        opt = row[3] if len(row) > 3 else {}
        t0 = 0.0 if k == 0 else snap(t(at))
        t1 = snap(t(rows[k + 1][0])) if k + 1 < len(rows) else end
        n = int(round(t1 * fps)) - int(round(t0 * fps))
        if n < 2:
            problems.append(f"shot {k} ({key} @ {at}) shorter than 2 frames")
        s = dict(src=to_out(sources[key]), t0=round(t0, 4), t1=round(t1, 4), **{"in": t_in})
        for p in PASS:
            if p in opt:
                s[p] = opt[p]
        s.setdefault("grade", spec.get("grade", "base"))
        shots.append(s)
        used = consumed(s.get("speed", 1.0), n, fps)
        lo, hi = (t_in - used, t_in) if s.get("reverse") else (t_in, t_in + used)
        if lo < 0 or hi > info[key]["dur"] - 0.05:
            problems.append(f"shot {k} ({key} {lo:.2f}-{hi:.2f}) outside the clip "
                            f"(0-{info[key]['dur']:.2f})")
        boxes = spec.get("keepout", {}).get(key, []) + opt.get("keepout", [])
        for u in np.linspace(0, 1, 9):
            win = window(kf(s.get("zoom", 1.0), u), kf(s.get("focus", 0.5), u), kf(s.get("fy", 0.5), u))
            bad = [b for b in boxes if hits(win, b)]
            if bad:
                problems.append(f"shot {k} ({key} @ {at}) shows keep-out {bad[0]} at u={u:.2f} "
                                f"(window {', '.join(f'{v:.3f}' for v in win)})")
                break
        scene = "?"
        for a0, a1 in scenes.get(key, []):
            if a0 - 0.02 <= lo and hi <= a1 + 0.02:
                scene = f"{a0:.1f}-{a1:.1f}"
                break
        else:
            if key in scenes:
                scene = "CROSSES A CUT"
                problems.append(f"shot {k} ({key} {lo:.2f}-{hi:.2f}) crosses a source scene cut")
        table.append(f"{k:2d} {str(at):>6} {t0:6.3f}-{t1:6.3f} {n:3d}f  {key:8s} "
                     f"{lo:7.2f}-{hi:7.2f}  scene {scene:13s} {s['grade']:6s} "
                     f"{' '.join(x if isinstance(x, str) else x['type'] for x in opt.get('fx', []))}")
        for item in opt.get("fx", []):
            if isinstance(item, dict):
                f = dict(item)
                if "u0" in f:
                    u0, u1 = f.pop("u0"), f.pop("u1", 1.0)
                    f["t"], f["dur"] = t0 + u0 * (t1 - t0), (u1 - u0) * (t1 - t0)
                else:
                    f["t"] = t(f["t"])
                    if "t_end" in f:
                        f["dur"] = t(f.pop("t_end")) - f["t"]
                fx.append(f)
            else:
                kind, _, amt = item.partition(":")
                fx += fx_macro(kind, float(amt) if amt else None, t0, t1, fps)
    for f in spec.get("fx", []):
        f = dict(f)
        f["t"] = t(f["t"])
        if "t_end" in f:
            f["dur"] = t(f.pop("t_end")) - f["t"]
        fx.append(f)
    for f in fx:
        f["t"] = round(max(0.0, f["t"]), 4)
        f["dur"] = round(f.get("dur", 0.1), 4)
    tl = dict(fps=fps, width=spec.get("width", 1920), height=spec.get("height", 1080),
              music=to_out(rel(spec["music"])), music_start=round(t_start, 4),
              duration=round(end, 4), fade_in=spec.get("fade_in", 0.02),
              fade_out=spec.get("fade_out", 0.3), grain=spec.get("grain", 0.04),
              loudness=spec.get("loudness", -10.0), true_peak=spec.get("true_peak", -1.0),
              crf=spec.get("crf", 17), maxrate=spec.get("maxrate", "20M"),
              bufsize=spec.get("bufsize", "40M"), src_max_h=spec.get("src_max_h", 1620),
              shots=shots, fx=sorted(fx, key=lambda f: f["t"]))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(tl, fh, indent=1)
    lens = np.diff([s["t0"] for s in shots] + [end])
    print("\n".join(table))
    print(f"{os.path.basename(a.out)}: {len(shots)} shots, {end:.2f}s "
          f"(music {t_start:.3f}-{t_start + end:.3f}s), shot length {lens.min():.2f}-{lens.max():.2f}s "
          f"(median {np.median(lens):.2f}), {len(fx)} fx")
    for p in problems:
        print("  PROBLEM:", p)
    if any("outside" in p or "2 frames" in p for p in problems):
        sys.exit(1)


if __name__ == "__main__":
    main()
