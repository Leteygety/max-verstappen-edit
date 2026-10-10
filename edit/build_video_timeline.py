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
shot). Free-form fx at the top level use beat refs; an "overlay" fx (title layer)
names its RGBA image in "src", relative to the cut list.

"transitions": "auto" hands the cuts to transitions.py: it measures the first
and last ~0.3 s of every shot (optical flow: pan direction, approach/zoom,
brightness, subject position) and weighs every cut by the music ("marks":
"drops", "build" and "silence" beat ranges, "hits" weighed like phrase starts,
"phrase" length; bass strength read from the track), then picks the transition
(whip along the shared motion, zoomtrans on approaches, smeardip on fast passes,
whiteout into brighter shots / phrase starts, dip, scanline break-up, two-shot
bandmix / strobecut / boxinvert in builds, lumamix in silences, flashcut on
drops, plain cuts off the beat), adds accents (punch / shake / RGB by weight),
moves shots without a fixed "focus" so the subject lands where the previous
one was (eye-trace, never into a keep-out box) and ramps shots without an
explicit speed into the big hits. A transition macro in a shot's own fx (or a
tail macro such as white / fadeout / glitchend on the shot before) keeps that
cut manual; "accents": false on a shot drops its accents; "kinds" ({source:
"car" | "max" | ...}) flags three shots of one kind in a row. The decisions are
printed as a table (cut, class, bass, motion out / in, luma, choice, reason).

Transition macros: cut, dip, black:N, whip, zoomtrans, whiteout, flashcut,
scanline, smeardip, bandmix, strobecut, boxinvert, lumamix, crossfade. Looks:
flash, white (tail blow-out), bloom, punch, shake, shakehit, rgb, glitch,
glitchend, strobe, invert, flicker, smearend, blocksend, ripple, mirror:S,
freeze:S, desatpulse:P, radialpulse:P, leak, fadein, fadeout.

Every shot is checked against the source: the consumed source range must fit
in the clip and should not cross a scene cut found by scan_motion.py
(media/work/scan/<clip>.json, if present), and the reframed window (zoom,
focus, fy) must stay clear of the burned-in graphics listed under "keepout"
(per source, plus per-shot "keepout" boxes; [x0, y0, x1, y1] fractions).

Usage: build_video_timeline.py edit/cuts_video/v1.json edit/timeline_v1_16x9.json [--no-auto]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fx_macro(kind, amt, t0, t1, fps, pd=None, p=None):
    """Cut effect for a shot running t0..t1 (edit seconds); pd = length of the
    shot before the cut (for transitions that start in the outgoing shot), p =
    planner parameters (angle, subject centre)."""
    f = 1.0 / fps
    d = t1 - t0
    pd = pd if pd is not None else d
    p = p or {}
    # ---- transitions (the cut into this shot sits at t0)
    if kind == "cut":
        return []
    if kind == "whiteout":  # outgoing blows out to white, incoming develops out of it
        w1, w2 = min(0.22, 0.45 * pd), min(0.18, 0.4 * d)
        # the picture stays readable through the haze (it never sits on flat white)
        return [dict(type="exposure", t=t0 - w1, dur=w1, amt=1.7, env="in"),
                dict(type="bloom", t=t0 - w1, dur=w1, amt=0.8, env="in"),
                dict(type="exposure", t=t0, dur=w2, amt=1.5, env="out"),
                dict(type="bloom", t=t0, dur=w2 * 1.6, amt=0.7, env="out"),
                dict(type="leak", t=t0 - w1, dur=w1 + w2, amt=0.45, env="tri", color="white")]
    if kind == "flashcut":
        return [dict(type="flash", t=t0, dur=0.12, amt=amt or 0.95, env="decay"),
                dict(type="bloom", t=t0, dur=min(0.5, d), amt=0.6, env="decay")]
    if kind == "scanline":  # break up into black bands and come back out of them
        w1, w2 = min(0.13, 0.4 * pd), min(0.1, 0.35 * d)
        return [dict(type="scanline", t=t0 - w1, dur=w1, amt=amt or 1.0, env="in"),
                dict(type="scanline", t=t0, dur=w2, amt=amt or 1.0, env="out")]
    if kind == "smeardip":  # fast pass: pan smear + pixel stretch into black
        w = min(0.16, 0.45 * pd)
        ang = p.get("angle", 0.0)
        return [dict(type="dirblur", t=t0 - w, dur=w, amt=0.9, angle=ang, env="in"),
                dict(type="smear", t=t0 - w * 0.6, dur=w * 0.6, amt=0.8, env="in"),
                dict(type="dip", t=t0 - w * 0.5, dur=w * 0.5, amt=0.95, env="in"),
                dict(type="dip", t=t0, dur=2 * f, amt=0.9, env="out")]
    if kind == "bandmix":  # two shots interleaved in bands, a few negative
        w = min(0.2, 0.5 * d)
        return [dict(type="bandmix", t=t0, dur=w, amt=1.0, env="hold"),
                dict(type="rgb", t=t0, dur=w, amt=1.0, jitter=True, env="hold")]
    if kind == "strobecut":  # A/B alternating frame by frame
        return [dict(type="strobecut", t=t0, dur=min(0.23, 0.6 * d), amt=1.0, env="hold")]
    if kind == "lumamix":
        return [dict(type="lumamix", t=t0, dur=min(0.5, 0.6 * d), amt=1.0, env="hold")]
    if kind == "crossfade":
        return [dict(type="crossfade", t=t0, dur=min(0.3, 0.5 * d), amt=1.0, env="hold")]
    if kind == "boxinvert":  # boxed negative around the subject + pixel stretch
        w = min(0.13, 0.4 * d)
        return [dict(type="boxinvert", t=t0, dur=w, amt=1.0, env="hold",
                     cx=p.get("cx", 0.5), cy=p.get("cy", 0.5)),
                dict(type="smear", t=t0, dur=w, amt=0.6, env="hold")]
    if kind == "shakehit":  # short hit on a bar line (lighter than "shake")
        return [dict(type="shake", t=t0, dur=min(0.3, d), amt=amt or 0.3, freq=1.8, env="decay")]
    # ---- looks inside a shot
    if kind == "flicker":  # irregular light flicker over the whole shot
        return [dict(type="flicker", t=t0, dur=d, amt=amt or 0.8, env="hold")]
    if kind == "smearend":
        w = min(amt or 0.12, d)
        return [dict(type="smear", t=t1 - w, dur=w, amt=0.9, env="in")]
    if kind == "blocksend":  # block break-up into black
        w = min(amt or 0.12, d)
        return [dict(type="blocks", t=t1 - w, dur=w, amt=1.0, env="in"),
                dict(type="dip", t=t1 - w * 0.5, dur=w * 0.5, amt=0.9, env="in")]
    if kind == "ripple":
        return [dict(type="wave", t=t0, dur=d, amt=amt or 0.8, mode="ripple", env="tri")]
    if kind == "mirror":   # mirrored panels for a flash of amt seconds (default 3 frames)
        return [dict(type="mirror", t=t0, dur=min(amt or 3 * f, d), amt=1.0, mode="tri", env="hold")]
    if kind == "freeze":  # hold the first frame for amt seconds, then play on
        return [dict(type="freeze", t=t0, dur=min(amt or 0.15, d), amt=1.0, env="hold")]
    if kind == "desatpulse":  # colour <-> monochrome on every beat
        return [dict(type="desat", t=t0, dur=d, amt=1.0, env="hold", pulse=amt or 0.5)]
    if kind == "radialpulse":  # radial blur pulses on 1/8 notes (amt = pulse period s)
        return [dict(type="zoomblur", t=t0, dur=d, amt=0.9, env="hold", pulse=amt or 0.25)]
    if kind == "leak":
        return [dict(type="leak", t=t0, dur=d, amt=amt or 0.7, env="tri", color="warm")]
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
    if kind == "whip":     # centred on the cut into this shot, along the shots' motion
        return [dict(type="whip", t=t0 - 0.1, dur=0.2, amt=amt or p.get("amt", 1.0),
                     angle=p.get("angle", 0.0))]
    if kind == "zoomtrans":
        return [dict(type="zoomtrans", t=t0 - 0.12, dur=0.24, amt=amt or p.get("amt", 1.0))]
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


def fx_kinds(opt):
    return [x.partition(":")[0] if isinstance(x, str) else x.get("type") for x in opt.get("fx", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cuts")
    ap.add_argument("out")
    ap.add_argument("--no-auto", action="store_true", help="ignore \"transitions\": \"auto\"")
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

    # ---- pass 1: shots on the grid
    plan_rows = []
    for k, row in enumerate(rows):
        at, key, t_in = row[0], row[1], float(row[2])
        opt = dict(row[3]) if len(row) > 3 else {}
        t0 = 0.0 if k == 0 else snap(t(at))
        t1 = snap(t(rows[k + 1][0])) if k + 1 < len(rows) else end
        n = int(round(t1 * fps)) - int(round(t0 * fps))
        beat = (float(at[1:]) if isinstance(at, str)
                else (t_start + float(at) - phase) / period)
        plan_rows.append(dict(at=at, key=key, t_in=t_in, opt=opt, t0=t0, t1=t1, n=n, beat=beat))

    def fits(r, speed):
        used = consumed(speed, r["n"], fps)
        lo, hi = (r["t_in"] - used, r["t_in"]) if r["opt"].get("reverse") else (r["t_in"], r["t_in"] + used)
        if lo < 0 or hi > info[r["key"]]["dur"] - 0.05:
            return False
        for a0, a1 in scenes.get(r["key"], []):
            if a0 - 0.02 <= lo and hi <= a1 + 0.02:
                return True
        return r["key"] not in scenes

    # ---- pass 2: shot-to-shot logic (transitions, accents, eye-trace, speed ramps)
    decisions = [None] * len(plan_rows)
    auto = spec.get("transitions") == "auto" and not a.no_auto
    notes = []
    if auto:
        import transitions as tr
        cache = os.path.join(root, "media", "work", "edges")
        cuts = []
        for r in plan_rows:
            o = r["opt"]
            edges = tr.shot_edges(sources[r["key"]], r["t_in"], o.get("speed", 1.0), max(2, r["n"]),
                                  fps, o.get("reverse", False), o.get("zoom", 1.0), cache)
            prev_tail = set(fx_kinds(plan_rows[len(cuts) - 1]["opt"])) & tr.TAIL if cuts else set()
            cuts.append(dict(beat=r["beat"], dur=r["t1"] - r["t0"], edges=edges,
                             manual=bool(set(fx_kinds(o)) & tr.MANUAL or prev_tail),
                             no_accents=o.get("accents") is False))
        marks = spec.get("marks", {})
        bass = tr.bass_strength(rel(spec["music"]), [t_start + r["t0"] for r in plan_rows],
                                (t_start, t_start + end)) if os.path.exists(rel(spec["music"])) else None
        decisions = tr.plan(cuts, marks, bass, fps)
        for k in range(1, len(plan_rows)):
            A, B, dcs = plan_rows[k - 1], plan_rows[k], decisions[k]
            # eye-trace: incoming subject where the outgoing one was (unless focus is fixed)
            if B["opt"].get("focus", "auto") == "auto":
                za = A["opt"].get("zoom", 1.0)
                fa = A["opt"].get("focus", 0.5)
                cx = tr.eye_trace(cuts[k - 1]["edges"], kf(za, 1.0), kf(fa if fa != "auto" else 0.5, 1.0),
                                  cuts[k]["edges"], kf(B["opt"].get("zoom", 1.0), 0.0))
                boxes = spec.get("keepout", {}).get(B["key"], []) + B["opt"].get("keepout", [])
                if cx is not None and any(
                        hits(window(kf(B["opt"].get("zoom", 1.0), u), cx, kf(B["opt"].get("fy", 0.5), u)), bx)
                        for u in np.linspace(0, 1, 9) for bx in boxes):
                    cx = None  # would show burned-in graphics: keep the centre
                if cx is not None:
                    B["opt"]["focus"] = cx
                    dcs["eyetrace"] = cx
                else:
                    B["opt"].pop("focus", None)
            # speed ramps into the big hits: accelerate, then brake right before the cut
            if dcs["cls"] in ("drop", "phrase") and "speed" not in A["opt"] \
                    and cuts[k - 1]["edges"]["tail"]["speed"] > 0.35 and A["t1"] - A["t0"] >= 0.4:
                ramp = [[0, 1.0], [0.5, 1.5], [0.78, 1.5], [1, 0.45]]
                if fits(A, ramp):
                    A["opt"]["speed"], A["opt"]["mblur"] = ramp, A["opt"].get("mblur", 1.2)
                    dcs["ramp_in"] = True
            if dcs["cls"] == "drop" and "speed" not in B["opt"] \
                    and cuts[k]["edges"]["head"]["speed"] > 0.35:
                land = [[0, 0.55], [0.3, 1.2], [1, 1.0]]
                if fits(B, land):
                    B["opt"]["speed"] = land
                    dcs["landing"] = True
        kinds = spec.get("kinds", {})
        run = 1
        for k in range(1, len(plan_rows)):
            ka, kb = kinds.get(plan_rows[k - 1]["key"]), kinds.get(plan_rows[k]["key"])
            run = run + 1 if ka and ka == kb else 1
            if run == 3:
                notes.append(f"shots {k - 2}-{k}: three '{kb}' shots in a row")

    # ---- pass 3: timeline shots, checks, fx
    shots, fx, problems, table = [], [], [], []
    for k, r in enumerate(plan_rows):
        at, key, t_in, opt, t0, t1, n = r["at"], r["key"], r["t_in"], r["opt"], r["t0"], r["t1"], r["n"]
        if n < 2:
            problems.append(f"shot {k} ({key} @ {at}) shorter than 2 frames")
        s = dict(src=to_out(sources[key]), t0=round(t0, 4), t1=round(t1, 4), **{"in": t_in})
        for p in PASS:
            if p in opt and opt[p] != "auto":
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
        pd = (t0 - plan_rows[k - 1]["t0"]) if k else None
        items = list(opt.get("fx", []))
        dcs = decisions[k]
        if dcs:
            if dcs["kind"] not in ("manual", "cut"):
                fx += fx_macro(dcs["kind"], None, t0, t1, fps, pd, dcs["params"])
            have = set(fx_kinds(opt))
            items += [x for x in dcs["accents"] if x.partition(":")[0] not in have
                      and not (x.startswith("shakehit") and "shake" in have)]
        table.append(f"{k:2d} {str(at):>6} {t0:6.3f}-{t1:6.3f} {n:3d}f  {key:8s} "
                     f"{lo:7.2f}-{hi:7.2f}  scene {scene:13s} {s['grade']:6s} "
                     f"{' '.join(x if isinstance(x, str) else x['type'] for x in items)}")
        for item in items:
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
                fx += fx_macro(kind, float(amt) if amt else None, t0, t1, fps, pd)
    for f in spec.get("fx", []):
        f = dict(f)
        if "src" in f:  # overlay layers: path relative to the timeline, like shot sources
            f["src"] = to_out(rel(f["src"]))
        f["t"] = t(f["t"])
        if "t_end" in f:
            f["dur"] = t(f.pop("t_end")) - f["t"]
        fx.append(f)
    for f in fx:
        f["t"] = round(max(0.0, f["t"]), 4)
        f["dur"] = round(f.get("dur", 0.1), 4)
    extra = {k: spec[k] for k in ("frame_blend", "sharpen", "halation") if k in spec}
    tl = dict(fps=fps, width=spec.get("width", 1920), height=spec.get("height", 1080),
              music=to_out(rel(spec["music"])), music_start=round(t_start, 4),
              duration=round(end, 4), fade_in=spec.get("fade_in", 0.02),
              fade_out=spec.get("fade_out", 0.3), grain=spec.get("grain", 0.04),
              loudness=spec.get("loudness", -10.0), true_peak=spec.get("true_peak", -1.0),
              crf=spec.get("crf", 17), maxrate=spec.get("maxrate", "20M"),
              bufsize=spec.get("bufsize", "40M"), src_max_h=spec.get("src_max_h", 1620),
              **extra, shots=shots, fx=sorted(fx, key=lambda f: f["t"]))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(tl, fh, indent=1)
    lens = np.diff([s["t0"] for s in shots] + [end])
    print("\n".join(table))
    if auto:
        print("\ncut  beat   class     bass  out-motion        in-motion         luma     -> transition (why) + accents")
        for k in range(1, len(plan_rows)):
            d, A, B = decisions[k], plan_rows[k - 1], plan_rows[k]
            ea, eb = cuts[k - 1]["edges"]["tail"], cuts[k]["edges"]["head"]
            mv = lambda e: f"{e['speed']:.2f}@{math.degrees(math.atan2(e['flow'][1], e['flow'][0])):+4.0f} z{e['div']:+.2f}"  # noqa: E731
            extra_s = "".join([f" eye->{d['eyetrace']}" if d.get("eyetrace") is not None else "",
                               " ramp-in" if d.get("ramp_in") else "", " landing" if d.get("landing") else ""])
            bs = f"{d['bass']:.2f}" if d["bass"] is not None else "  - "
            print(f"{k:3d} {str(B['at']):>6}  {d['cls']:9s} {bs}  {mv(ea):17s} {mv(eb):17s} "
                  f"{ea['luma']:.2f}>{eb['luma']:.2f}  -> {d['kind']} ({d['reason']})"
                  f"{' + ' + ' '.join(d['accents']) if d['accents'] else ''}{extra_s}")
        for nt in notes:
            print("  NOTE:", nt)
    print(f"{os.path.basename(a.out)}: {len(shots)} shots, {end:.2f}s "
          f"(music {t_start:.3f}-{t_start + end:.3f}s), shot length {lens.min():.2f}-{lens.max():.2f}s "
          f"(median {np.median(lens):.2f}), {len(fx)} fx")
    for p in problems:
        print("  PROBLEM:", p)
    if any("outside" in p or "2 frames" in p for p in problems):
        sys.exit(1)


if __name__ == "__main__":
    main()
