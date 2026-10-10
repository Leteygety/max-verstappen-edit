#!/usr/bin/env python3
"""Shot-to-shot logic for the video cut lists: measures what happens at every
cut and picks how to get from one shot to the next.

For every shot it reads the first and last ~0.3 s actually used (after speed
ramps / reverse) at low resolution and measures, with optical flow:
  flow      dominant screen motion (dx, dy in frame widths per output second)
  speed     its magnitude, div  zoom/approach (+ = expanding, a car coming at us)
  luma, sat brightness and colour
  subject   where the moving / saturated subject sits (x, y fractions)

Each cut gets a musical weight from the beat grid and the cut list's marks
(drop, build, silence, phrase length; bass strength from the track if it can
be read), and a transition is scored from both:

  whip       both sides move the same way      -> pan blur along that motion
  zoomtrans  either side zooms / approaches    -> radial punch through the cut
  smeardip   fast outgoing pass into a darker shot -> motion smear into black
  whiteout   much brighter incoming, or a phrase start -> blow out to white
  dip        the reference's 2-4 frame luma crash (static or darker incoming)
  scanline   static close-ups on bar lines      -> break-up into black bands
  bandmix / strobecut / boxinvert   builds and conflicting motion -> two-shot glitch
  lumamix    inside a silence                   -> highlights burn through
  flashcut   drops                               -> white frame + bloom
  cut        off-beat cuts stay clean (a 2-frame RGB kick at most)

Repeats are penalised (no transition three times running, whiteout at most
once per 8 beats, two-shot glitches at most once per 4 beats). On top come
accents scaled by the hit (punch / shake / RGB on drops and phrase starts,
lighter shake on bars, small pulses on the beats inside long shots), an
eye-trace correction (the incoming subject lands where the outgoing one was,
when the shot's focus is not fixed) and speed ramps into the big hits
(accelerate, then brake to ~0.45x right before the cut; a short slow-motion
landing after a drop) for shots without an explicit speed.

Used by build_video_timeline.py when the cut list has "transitions": "auto".
"""
import hashlib
import json
import math
import os
import subprocess

import cv2
import numpy as np

EDGE = 0.3          # seconds of output analysed at each end of a shot
RATE = 15.0         # analysis frames per second
AW = 192            # analysis width

# transition macros a shot's own fx list may already name (manual override)
MANUAL = {"dip", "black", "whip", "zoomtrans", "whiteout", "flash", "flashcut",
          "scanline", "smeardip", "bandmix", "strobecut", "lumamix", "crossfade",
          "boxinvert", "cut", "fadein"}
# tail macros on the outgoing shot already shape the cut out of it
TAIL = {"white", "fadeout", "glitchend", "smearend", "blocksend"}


# ---------------------------------------------------------------- footage

def _decode(path, t0, dur, w=AW):
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", path], capture_output=True, text=True).stdout)
    s = info["streams"][0]
    h = int(round(w * s["height"] / s["width"] / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t0):.3f}", "-t", f"{dur:.3f}",
                          "-i", path, "-an", "-vf", f"fps={RATE},scale={w}:{h}",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def measure(path, t0, t1, cache_dir=None):
    """Motion / light features of source range t0..t1 (seconds, t0 < t1)."""
    key = None
    if cache_dir:
        key = hashlib.sha1(f"{os.path.abspath(path)}|{t0:.3f}|{t1:.3f}|v3".encode()).hexdigest()[:16]
        cp = os.path.join(cache_dir, key + ".json")
        if os.path.exists(cp):
            return json.load(open(cp))
    f = _decode(path, t0, max(t1 - t0, 2.5 / RATE))
    if len(f) < 2:
        res = dict(flow=[0.0, 0.0], speed=0.0, div=0.0, luma=0.4, sat=0.1, subject=[0.5, 0.5])
    else:
        h, w = f.shape[1:3]
        gray = [cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (0, 0), 1.0) for x in f]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rx, ry = xx - w / 2, yy - h / 2
        rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-3
        flows, objs, divs, sal = [], [], [], np.zeros((h, w), np.float32)
        for a, b in zip(gray[:-1], gray[1:]):
            fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 13, 3, 5, 1.1, 0)
            med = np.median(fl.reshape(-1, 2), axis=0)   # camera / pan
            flows.append(med)
            rel = fl - med
            mag = np.linalg.norm(rel, axis=2)
            sel = mag > max(np.percentile(mag, 88), 0.4)  # the moving subject (a passing car)
            objs.append(fl[sel].mean(0) if sel.sum() > 20 else med)
            # expansion: flow pointing away from the centre, relative to the pan
            divs.append(float(np.mean((rel[..., 0] * rx + rel[..., 1] * ry) / rn) / w))
            sal += np.linalg.norm(rel, axis=2)
        cam = np.median(np.array(flows), axis=0) * RATE / w   # frame widths per second
        obj = np.median(np.array(objs), axis=0) * RATE / w
        # the motion the eye follows: the pan if there is one, else the moving subject
        # (weighted down: it covers only part of the frame)
        fd = cam if np.hypot(*cam) > 0.5 * np.hypot(*obj) * 0.6 else obj * 0.6
        rgb = f.astype(np.float32) / 255
        satmap = (rgb.max(-1) - rgb.min(-1)).mean(0)
        sal = sal / (sal.max() + 1e-6) + 0.5 * satmap / (satmap.max() + 1e-6)
        sal = cv2.GaussianBlur(sal, (0, 0), 4)
        sal = np.maximum(sal - np.percentile(sal, 70), 0)
        tot = sal.sum() + 1e-6
        res = dict(flow=[float(fd[0]), float(fd[1])], speed=float(np.hypot(*fd)),
                   cam=float(np.hypot(*cam)), obj=float(np.hypot(*obj)),
                   div=float(np.median(divs) * RATE), luma=float(rgb.mean()),
                   sat=float(satmap.mean()),
                   subject=[float((sal * xx).sum() / tot / w), float((sal * yy).sum() / tot / h)])
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(res, open(os.path.join(cache_dir, key + ".json"), "w"))
    return res


def offsets(speed, n, fps, reverse=False):
    """render.py's time remap: source offset per output frame (n+1 values)."""
    u = np.arange(n + 1) / max(1, n)
    if isinstance(speed, list):
        sp = np.interp(u, [p[0] for p in speed], [p[1] for p in speed])
    else:
        sp = np.full(n + 1, float(speed))
    off = np.concatenate([[0], np.cumsum((sp[1:] + sp[:-1]) / 2) / fps])
    if reverse:
        off = off[-1] - off
    return off, sp


def shot_edges(path, t_in, speed, n, fps, reverse, zoom, cache_dir):
    """Head and tail features of a shot as it plays (speed and direction applied)."""
    off, sp = offsets(speed, n, fps, reverse)
    k = max(2, min(n, int(round(EDGE * fps))))
    out = {}
    for side, idx in (("head", slice(0, k + 1)), ("tail", slice(n - k, n + 1))):
        o = off[idx]
        lo, hi = t_in + o.min(), t_in + o.max()
        if hi - lo < 1.0 / RATE:  # freeze / very slow: look at a short window anyway
            hi = lo + 2.0 / RATE
        m = dict(measure(path, lo, hi, cache_dir))
        v = float(np.mean(np.abs(sp[idx])))
        sgn = -1.0 if (o[-1] < o[0]) else 1.0
        m["flow"] = [m["flow"][0] * v * sgn, m["flow"][1] * v * sgn]
        m["speed"] = m["speed"] * v
        m["div"] = m["div"] * v * sgn
        # digital zoom in the cut list reads as expansion too
        z = zoom if isinstance(zoom, list) else [[0, zoom], [1, zoom]]
        zu = lambda u: float(np.interp(u, [p[0] for p in z], [p[1] for p in z]))  # noqa: E731
        span = k / max(1, n)
        u0, u1 = (0.0, span) if side == "head" else (1 - span, 1.0)
        m["div"] += (math.log(zu(u1)) - math.log(zu(u0))) / (k / fps)
        out[side] = m
    return out


# ---------------------------------------------------------------- music

def bass_strength(music, t_song, window):
    """Low-band onset strength (0..1) at song times, normalised over the window."""
    try:
        import librosa
    except ImportError:
        return None
    try:
        a, b = window
        y, sr = librosa.load(music, sr=22050, mono=True, offset=max(0.0, a - 0.5), duration=b - a + 1.0)
    except Exception:  # noqa: BLE001 - unreadable / missing track: fall back to the grid
        return None
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[fr < 150].sum(0)
    env = np.maximum(0, np.diff(low, prepend=low[0]))
    tt = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop) + max(0.0, a - 0.5)
    ref = np.percentile(env, 97) + 1e-9
    out = []
    for t in t_song:
        m = (tt > t - 0.045) & (tt < t + 0.045)
        out.append(float(min(1.0, env[m].max() / ref)) if m.any() else 0.0)
    return out


def classify(beat, marks):
    """Musical weight of a cut on song beat `beat` (float)."""
    near = lambda x, y: abs(x - y) < 0.02  # noqa: E731
    b = lambda v: float(str(v).lstrip("b"))  # noqa: E731
    for r in marks.get("silence", []):
        if b(r[0]) <= beat < b(r[1]):
            return "silence"
    if any(near(beat, b(d)) for d in marks.get("drops", [])):
        return "drop"
    if any(near(beat, b(h)) for h in marks.get("hits", [])):
        return "phrase"
    build = any(b(r[0]) <= beat < b(r[1]) for r in marks.get("build", []))
    frac = beat - math.floor(beat + 1e-6)
    if frac > 0.02:
        return "build-off" if build else "off"
    if build:
        return "build"
    origin = b(marks.get("phrase_origin", marks.get("drops", ["b0"])[0] if marks.get("drops") else "b0"))
    rel = int(round(beat - origin))
    if rel % int(marks.get("phrase", 8)) == 0:
        return "phrase"
    if rel % 4 == 0:
        return "bar"
    return "beat"


WEIGHT = {"drop": 1.0, "phrase": 0.8, "bar": 0.6, "build": 0.55, "beat": 0.45,
          "build-off": 0.35, "off": 0.25, "silence": 0.15}


# ---------------------------------------------------------------- choice

def _angle(v):
    return math.degrees(math.atan2(v[1], v[0]))


def _adiff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def choose(cls, strength, A, B, dA, dB, history, beat, last_at):
    """Score every transition for the cut A -> B; return (kind, params, reason)."""
    a, b = A["tail"], B["head"]
    fast = 0.35   # frame widths per second
    sc, why, prm = {}, {}, {}

    def put(kind, s, reason, **p):
        sc[kind], why[kind], prm[kind] = s, reason, p

    put("cut", 0.30 + (0.45 if cls in ("off", "build-off") else 0) + (0.2 if dB < 0.3 else 0)
        + (0.3 if b["speed"] > fast and cls in ("beat", "bar") else 0),
        "off-beat / short shot / the incoming motion carries the cut")
    put("dip", 0.50 + (0.25 if a["luma"] > b["luma"] + 0.08 else 0)
        + (0.15 if a["speed"] < fast and b["speed"] < fast else 0)
        - (0.5 if cls in ("off", "build-off") else 0), "luma crash (signature)")
    # motion continuity: both sides move the same way -> pan along it
    if a["speed"] > fast and b["speed"] > fast:
        d = _adiff(_angle(a["flow"]), _angle(b["flow"]))
        if d < 60:
            ang = _angle([a["flow"][0] + b["flow"][0], a["flow"][1] + b["flow"][1]])
            put("whip", 0.75 + 0.35 * (1 - d / 60) + 0.1 * min(a["speed"], 2),
                f"both move {ang:+.0f} deg (diff {d:.0f})", angle=round(ang, 1),
                amt=round(min(1.2, 0.6 + 0.3 * (a["speed"] + b["speed"]) / 2), 2))
        else:
            put("bandmix", 0.55 + (0.2 if cls in ("build", "build-off", "phrase") else 0),
                f"opposing motion ({d:.0f} deg apart): hide it in a band glitch")
    elif a["speed"] > fast * 1.6:
        ang = _angle(a["flow"])
        put("whip", 0.55, f"outgoing pans {ang:+.0f} deg", angle=round(ang, 1), amt=0.8)
    if a["speed"] > fast * 1.4 and b["luma"] < a["luma"] + 0.05:
        put("smeardip", 0.5 + 0.1 * min(a["speed"], 2), "fast pass smears into black",
            angle=round(_angle(a["flow"]), 1))
    zoomy = max(a["div"], b["div"])
    if zoomy > 0.25:
        put("zoomtrans", 0.5 + min(0.4, zoomy * 0.5) + (0.2 if cls in ("drop", "phrase") else 0),
            f"approach / zoom (div {zoomy:.2f})", amt=round(min(1.2, 0.7 + zoomy * 0.3), 2))
    if b["luma"] > a["luma"] + 0.15 or cls in ("phrase",):
        put("whiteout", 0.45 + max(0.0, (b["luma"] - a["luma"]) * 1.5)
            + (0.15 if cls == "phrase" else 0), "brighter incoming / phrase start")
    if a["speed"] < fast and b["speed"] < fast and cls in ("bar", "phrase", "beat"):
        put("scanline", 0.5 + (0.1 if cls != "beat" else 0), "static close-ups: band break-up")
    if cls in ("build", "build-off"):
        put("strobecut", 0.6 + (0.2 if dB < 0.4 else 0), "build: A/B strobe")
        put("bandmix", max(sc.get("bandmix", 0), 0.55), "build: band glitch")
        put("boxinvert", 0.5, "build: boxed negative", cx=round(b["subject"][0], 3),
            cy=round(b["subject"][1], 3))
    if cls == "silence":
        put("lumamix", 1.2, "silence: highlights burn through")
        sc["cut"] -= 0.4
    if cls == "drop":
        put("flashcut", 1.3, "drop: white frame + bloom")
    # variety: no transition three times running, rare ones stay rare
    for kind in sc:
        if history and history[-1] == kind:
            sc[kind] -= 0.35
        if len(history) > 1 and history[-2] == kind:
            sc[kind] -= 0.15
    if beat - last_at.get("whiteout", -99) < 8:
        sc["whiteout"] = sc.get("whiteout", 0) - 0.6
    for k in ("bandmix", "strobecut", "boxinvert"):
        if k in sc and beat - last_at.get("glitch2", -99) < 4:
            sc[k] -= 0.5
    if "scanline" in sc and beat - last_at.get("scanline", -99) < 3:
        sc["scanline"] -= 0.5
    best = max(sc, key=sc.get)
    return best, prm.get(best, {}), why[best]


def accents(cls, bass, B, dB):
    """Extra hits on the incoming shot, scaled by the cut's weight. Shots with their
    own motion get less: the movement already carries the beat."""
    s = bass if bass is not None else WEIGHT[cls]
    still = B["head"]["speed"] < 0.2
    out = []
    if cls == "drop":
        out += [f"punch:{0.8 + 0.2 * s:.2f}", f"shake:{0.7 + 0.3 * s:.2f}", "rgb:1.0"]
    elif cls == "phrase":
        out += [f"punch:{0.45 + 0.25 * s:.2f}", f"shake:{0.35 + 0.35 * s:.2f}"]
    elif cls == "bar":
        out += [f"shakehit:{(0.25 if still else 0.12) + 0.2 * s:.2f}"]
        if still:
            out += [f"punch:{0.2 + 0.2 * s:.2f}"]
    elif cls == "beat" and still and s > 0.6:
        out += [f"shakehit:{0.1 + 0.15 * s:.2f}"]
    elif cls == "build-off":
        out += ["rgb:0.8"]
    return out


def plan(cuts, marks, bass, fps):
    """cuts: list of dicts per shot (beat, dur, edges, manual, cls...). Returns
    one decision per shot (index 0 = the opening shot, no transition)."""
    history, last_at, out = [], {}, [None]
    for k in range(1, len(cuts)):
        A, B = cuts[k - 1], cuts[k]
        cls = classify(B["beat"], marks) if B["beat"] is not None else "beat"
        s = bass[k] if bass else None
        if B["manual"]:
            kind, prm, why = "manual", {}, "cut list sets the transition"
        else:
            kind, prm, why = choose(cls, s, A["edges"], B["edges"], A["dur"], B["dur"],
                                    history, B["beat"] or 0.0, last_at)
            history.append(kind)
            if kind == "whiteout":
                last_at["whiteout"] = B["beat"] or 0.0
            if kind in ("bandmix", "strobecut", "boxinvert"):
                last_at["glitch2"] = B["beat"] or 0.0
            if kind == "scanline":
                last_at["scanline"] = B["beat"] or 0.0
        acc = [] if B.get("no_accents") else accents(cls, s, B["edges"], B["dur"])
        # light hits on plain beats alternate (a shake on every cut turns into noise)
        if cls == "beat" and any(x.startswith("shakehit") for x in acc):
            if (B["beat"] or 0.0) - last_at.get("shakehit", -99) < 1.9:
                acc = [x for x in acc if not x.startswith("shakehit")]
            else:
                last_at["shakehit"] = B["beat"] or 0.0
        out.append(dict(kind=kind, params=prm, reason=why, cls=cls, bass=s, accents=acc))
    return out


def eye_trace(A_edges, A_zoom_end, A_focus_end, B_edges, B_zoom0):
    """Window centre for B so its subject lands where A's subject was on screen."""
    if B_zoom0 < 1.02:
        return None
    hw_a = 0.5 / max(A_zoom_end, 1.0)
    cxa = min(max(A_focus_end, hw_a), 1 - hw_a)
    sx = 0.5 + (A_edges["tail"]["subject"][0] - cxa) * A_zoom_end  # screen x of A's subject
    hw = 0.5 / B_zoom0
    target = B_edges["head"]["subject"][0] - (sx - 0.5) / B_zoom0
    cx = 0.5 + 0.7 * (target - 0.5)  # most of the way: a full jump to the edge reads as a pan
    return round(min(max(cx, hw), 1 - hw), 3)
