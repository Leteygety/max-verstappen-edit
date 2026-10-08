#!/usr/bin/env python3
"""Render a beat-synced vertical edit from a timeline JSON.

Every output frame is built in one pass: source frames are time-remapped
(speed ramps, freezes, reverse, stutter), reframed to 9:16 with a single
affine resample (focus pan + zoom + shake + rotation), graded, run through
the effect stack and piped straight into the final encoder, so the footage
is decoded once and encoded once.

Time fields in the timeline accept seconds (1.25), beat positions ("b12",
"b12.5" -- fractional beats interpolate the grid) or bass hits ("h7").

Usage: render.py timeline.json out.mp4 [--preview] [--from S] [--to S]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from numba import njit, prange

cv2.setNumThreads(os.cpu_count() or 4)


# ---------------------------------------------------------------- time refs

class Clock:
    def __init__(self, beats_path):
        self.beats = self.hits = None
        if beats_path:
            with open(beats_path) as fh:
                b = json.load(fh)
            self.beats = np.array(b["beats"])
            self.hits = np.array([h[0] for h in b["bass_hits"]])

    def __call__(self, v):
        if v is None or isinstance(v, (int, float)):
            return v
        v = v.strip()
        if v[0] == "b":
            x = float(v[1:])
            i = int(math.floor(x))
            i = max(0, min(i, len(self.beats) - 2))
            return float(self.beats[i] + (x - i) * (self.beats[i + 1] - self.beats[i]))
        if v[0] == "h":
            return float(self.hits[int(v[1:])])
        return float(v)


def kf(spec, u):
    """Evaluate a keyframe list [[u, value], ...] (or a scalar) at u in 0..1."""
    if not isinstance(spec, list):
        return spec
    if not isinstance(spec[0], list):
        return spec  # plain vector, e.g. [x, y]
    us = [p[0] for p in spec]
    vs = [p[1] for p in spec]
    return float(np.interp(u, us, vs))


def smooth(u):
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- sources

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,color_space", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(n) / float(d), s.get("color_space")


def is_still(path):
    return os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_frames(path, start, length, crop, scale_h=None, interp=None):
    """Decode [start, start+length) of a source, cropped to crop=(x, y, w, h)."""
    w0, h0, fps, cs = probe(path)
    x, y, w, h = crop
    vf = [f"crop={w}:{h}:{x}:{y}"]
    if scale_h:
        sw = int(round(w * scale_h / h / 2) * 2)
        vf.append(f"scale={sw}:{scale_h}:flags=area")
        w, h = sw, scale_h
    if interp:
        # optical-flow interpolation for clean slow motion
        fps = fps * interp
        vf.append(f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    matrix = "bt709" if (cs in (None, "unknown", "bt709") and h0 >= 720) else "bt601"
    vf.append(f"scale=in_color_matrix={matrix}:in_range=tv:out_range=pc")
    # a still image is a single frame at t=0: decode it whole (sample() holds it)
    seek = [] if is_still(path) else ["-ss", f"{max(0, start):.4f}", "-t", f"{length:.4f}"]
    cmd = ["ffmpeg", "-v", "error", *seek,
           "-i", path, "-an", "-vf", ",".join(vf), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if len(frames) == 0:
        raise RuntimeError(f"no frames decoded from {path} @ {start:.2f}s")
    return frames, fps


def auto_focus(path, start, length):
    """Horizontal focus (0..1) from motion + Red Bull-ish saturated colour."""
    w0, h0, fps, _ = probe(path)
    sw, sh = 320, max(2, int(round(320 * h0 / w0 / 2) * 2))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
                          "-i", path, "-an", "-vf", f"fps=12,scale={sw}:{sh}", "-pix_fmt", "rgb24",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, sh, sw, 3).astype(np.float32) / 255
    if len(f) < 2:
        return 0.5
    motion = np.abs(np.diff(f, axis=0)).mean(-1).mean(0)
    sat = (f.max(-1) - f.min(-1)).mean(0)
    sal = cv2.GaussianBlur(motion / (motion.max() + 1e-6) + 0.6 * sat / (sat.max() + 1e-6),
                           (0, 0), 6)
    col = sal.sum(0)
    col -= col.min()
    xs = np.arange(sw)
    return float(np.clip((col * xs).sum() / (col.sum() + 1e-6) / sw, 0.0, 1.0))


# ---------------------------------------------------------------- shot prep

class Shot:
    def __init__(self, spec, clock, W, H, fps, preview):
        self.s = spec
        self.t0, self.t1 = clock(spec["t0"]), clock(spec["t1"])
        self.W, self.H, self.fps = W, H, fps
        n = max(1, int(round(self.t1 * fps)) - int(round(self.t0 * fps)))
        self.n = n
        # time remap: integrate the speed curve over output time
        speed = spec.get("speed", 1.0)
        u = (np.arange(n + 1)) / max(1, n)
        sp = np.array([kf(speed, x) for x in u])
        off = np.concatenate([[0], np.cumsum((sp[1:] + sp[:-1]) / 2) / fps])
        if spec.get("reverse"):
            off = off[-1] - off
        st = spec.get("stutter", 1)
        if st and st > 1:
            off = off[(np.arange(n + 1) // st) * st]
        self.off = off
        self.speed = sp
        self.src = spec["src"]
        sw, sh, sfps, _ = probe(self.src)
        self.sw, self.sh = sw, sh
        t_in = clock(spec.get("in", 0.0))
        lo, hi = t_in + off.min(), t_in + off.max()
        margin = 2.0 / sfps
        self.load_t = max(0.0, lo - margin)
        # focus window (horizontal pan); y focus used when zoomed in
        focus = spec.get("focus", 0.5)
        if focus == "auto":
            focus = auto_focus(self.src, lo, max(hi - lo, 0.3))
            spec["focus"] = round(focus, 3)
        self.focus = focus
        self.fy = spec.get("fy", 0.5)
        cw = min(sw, sh * W / H)  # 9:16 window at zoom 1 (full source height)
        self.cw, self.ch = cw, sh if cw < sw else sw * H / W
        fxs = [kf(focus, x) for x in np.linspace(0, 1, 9)]
        need = min(sw, int(math.ceil(cw)) + 16)
        x0 = int(np.clip(math.floor(min(fxs) * sw - cw / 2) - 8, 0, sw - need))
        x1 = int(np.clip(math.ceil(max(fxs) * sw + cw / 2) + 8, x0 + need, sw))
        x0 -= x0 % 2
        x1 -= (x1 - x0) % 2
        scale_h = None
        if preview and sh > H:
            scale_h = H + H % 2
        frames, self.sfps = load_frames(self.src, self.load_t, hi - self.load_t + margin,
                                        (x0, 0, x1 - x0, sh), scale_h, spec.get("interp"))
        self.k = frames.shape[1] / sh  # strip pixels per source pixel
        self.frames = frames
        self.x0 = x0
        self.t_in = t_in

    def sample(self, i):
        """Source frame for output frame i (0..n-1), motion-blurred when fast."""
        ts = self.t_in + self.off[i] - self.load_t
        idx = ts * self.sfps
        nf = len(self.frames)
        step = abs(self.off[min(i + 1, self.n)] - self.off[i]) * self.sfps
        blur = self.s.get("mblur", 1.0)
        taps = int(round(step * 0.75 * blur)) if step > 2.5 and blur > 0 else 1
        taps = min(taps, 16)
        if taps > 1:
            d = 1 if self.off[min(i + 1, self.n)] >= self.off[i] else -1
            ids = np.clip(np.round(idx + d * np.arange(taps) * (step * 0.75 * blur / taps)),
                          0, nf - 1).astype(int)
            acc = self.frames[ids].astype(np.float32).mean(0)
            return acc / 255.0
        a = int(math.floor(idx))
        fr = idx - a
        a = int(np.clip(a, 0, nf - 1))
        b = min(a + 1, nf - 1)
        if fr < 0.02 or a == b or self.s.get("interp") == "none":
            return self.frames[a].astype(np.float32) / 255.0
        return (self.frames[a].astype(np.float32) * (1 - fr)
                + self.frames[b].astype(np.float32) * fr) / 255.0

    def geometry(self, i):
        u = i / max(1, self.n - 1)
        zoom = kf(self.s.get("zoom", 1.0), u)
        fx = kf(self.focus, u)
        fy = kf(self.fy, u)
        return zoom, fx, fy, u


# ---------------------------------------------------------------- grading

GRADES = {
    # sat: base saturation, keep: saturation kept on Red Bull hues + skin,
    # con: contrast strength, pivot, cool: shadow blue push, warm: highlight warmth,
    # gain: exposure (stops), black: black point
    "base":   dict(sat=0.55, keep=0.9, con=5.0, pivot=0.42, cool=0.05, warm=0.04, gain=0.0, black=0.03),
    "cold":   dict(sat=0.35, keep=0.7, con=5.5, pivot=0.45, cool=0.08, warm=0.0, gain=-0.15, black=0.035),
    "warm":   dict(sat=0.7, keep=1.0, con=5.0, pivot=0.42, cool=0.03, warm=0.09, gain=0.1, black=0.03),
    "bw":     dict(sat=0.0, keep=0.12, con=6.5, pivot=0.45, cool=0.02, warm=0.0, gain=0.0, black=0.035),
    "teal":   dict(sat=1.05, keep=1.15, con=5.0, pivot=0.42, cool=0.12, warm=0.06, gain=0.1, black=0.02),
    "bleach": dict(sat=0.25, keep=0.5, con=3.5, pivot=0.35, cool=0.03, warm=0.05, gain=0.9, black=0.0),
    "night":  dict(sat=0.6, keep=1.0, con=6.0, pivot=0.4, cool=0.1, warm=0.06, gain=-0.1, black=0.03),
    "none":   None,
}


@njit(cache=True, fastmath=True, inline="always")
def _bump(d, width):
    v = 1.0 - abs(d) / width
    return v if v > 0.0 else 0.0


@njit(cache=True, fastmath=True, inline="always")
def _curve(v, black, con, pivot, s0, s1):
    v = (v - black) / (1.0 - black)
    if v < 0.0:
        v = 0.0
    if v > 0.85:
        v = v / (1.0 + (v - 0.85) * 1.5)
    if v > 1.2:
        v = 1.2
    return (1.0 / (1.0 + math.exp(-con * (v - pivot))) - s0) / (s1 - s0)


@njit(parallel=True, cache=True, fastmath=True)
def _grade_kernel(img, out, gain, sat, keep, con, pivot, cool, warm, black):
    H, W = img.shape[0], img.shape[1]
    s0 = 1.0 / (1.0 + math.exp(con * pivot))
    s1 = 1.0 / (1.0 + math.exp(-con * (1.0 - pivot)))
    for yy in prange(H):
        for xx in range(W):
            r = img[yy, xx, 0] * gain
            g = img[yy, xx, 1] * gain
            b = img[yy, xx, 2] * gain
            # hue / saturation of the exposed pixel
            rc, gc, bc = min(r, 1.0), min(g, 1.0), min(b, 1.0)
            mx = max(rc, gc, bc)
            mn = min(rc, gc, bc)
            d = mx - mn
            s = d / mx if mx > 1e-6 else 0.0
            h = 0.0
            if d > 1e-6:
                if mx == rc:
                    h = 60.0 * (((gc - bc) / d) % 6.0)
                elif mx == gc:
                    h = 60.0 * ((bc - rc) / d + 2.0)
                else:
                    h = 60.0 * ((rc - gc) / d + 4.0)
            # keep Red Bull red / blue / yellow and skin; desaturate the rest
            red = max(_bump(h, 25.0), _bump(h - 360.0, 25.0))
            k = max(red, _bump(h - 28.0, 18.0) * 0.8, _bump(h - 225.0, 30.0),
                    _bump(h - 52.0, 14.0)) * min(s * 2.5, 1.0)
            f = sat + (keep - sat) * k
            y = r * 0.2126 + g * 0.7152 + b * 0.0722
            r = y + (r - y) * f
            g = y + (g - y) * f
            b = y + (b - y) * f
            # black point, highlight shoulder, filmic S-curve
            o0 = _curve(r, black, con, pivot, s0, s1)
            o1 = _curve(g, black, con, pivot, s0, s1)
            o2 = _curve(b, black, con, pivot, s0, s1)
            # split tone: cool shadows, warm highlights
            y = o0 * 0.2126 + o1 * 0.7152 + o2 * 0.0722
            y = min(max(y, 0.0), 1.0)
            sh = (1.0 - y) * (1.0 - y) * cool
            hi = y * y * warm
            out[yy, xx, 0] = min(max(o0 - 0.6 * sh + 0.7 * hi, 0.0), 1.0)
            out[yy, xx, 1] = min(max(o1 - 0.1 * sh + 0.25 * hi, 0.0), 1.0)
            out[yy, xx, 2] = min(max(o2 + 0.7 * sh - 0.6 * hi, 0.0), 1.0)


def grade(img, name, extra_stops=0.0):
    g = GRADES.get(name or "base")
    if g is None:
        return img * np.float32(2 ** extra_stops) if extra_stops else img
    out = np.empty_like(img, dtype=np.float32)
    _grade_kernel(np.ascontiguousarray(img, dtype=np.float32), out,
                  2 ** float(g["gain"] + extra_stops), g["sat"], g["keep"], g["con"],
                  g["pivot"], g["cool"], g["warm"], g["black"])
    return out


# ---------------------------------------------------------------- effects

def envelope(fx, t, fps):
    t0, d = fx["_t"], max(fx["_d"], 0.5 / fps)
    u = (t - t0) / d
    if u < 0 or u >= 1:
        return 0.0, u
    shape = fx.get("env", "decay")
    if shape == "hold":
        e = 1.0
    elif shape == "decay":
        e = math.exp(-u * 4.0) * (1 - u ** 6)
    elif shape == "tri":
        e = 1 - abs(2 * u - 1)
    elif shape == "in":
        e = u * u
    elif shape == "out":
        e = (1 - u) ** 2
    else:
        e = 1.0
    return e, u


def noise1(t, seed, freqs=(3.1, 7.3, 13.7)):
    r = np.random.default_rng(seed)
    ph = r.uniform(0, 2 * math.pi, len(freqs))
    return sum(math.sin(2 * math.pi * f * t + p) / (k + 1) for k, (f, p) in
               enumerate(zip(freqs, ph))) / 1.8


def dir_blur(img, length, angle):
    length = int(length)
    if length < 3:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, m, (length, length))
    k /= k.sum()
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def zoom_blur(img, amt, cx=0.5, cy=0.5, taps=8):
    if amt < 0.005:
        return img
    H, W = img.shape[:2]
    acc = np.zeros_like(img)
    for j in range(taps):
        s = 1 + amt * j / (taps - 1)
        m = np.array([[s, 0, (1 - s) * cx * W], [0, s, (1 - s) * cy * H]], np.float32)
        acc += cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return acc / taps


def rgb_split(img, px, radial=0.0):
    if abs(px) < 0.5 and radial < 0.001:
        return img
    H, W = img.shape[:2]
    out = img.copy()
    for c, sgn in ((0, 1), (2, -1)):
        s = 1 + sgn * radial
        m = np.array([[s, 0, (1 - s) * W / 2 + sgn * px], [0, s, (1 - s) * H / 2]], np.float32)
        out[..., c] = cv2.warpAffine(np.ascontiguousarray(img[..., c]), m, (W, H),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def glitch(img, amt, rng, prev):
    H, W = img.shape[:2]
    out = img.copy()
    nb = int(3 + amt * 14)
    for _ in range(nb):
        h = int(rng.uniform(0.004, 0.06) * H)
        y = int(rng.uniform(0, H - h))
        shift = int(rng.normal(0, amt * W * 0.12))
        band = out[y:y + h]
        band = np.roll(band, shift, axis=1)
        if rng.random() < 0.5:
            cs = int(rng.normal(0, amt * W * 0.03))
            band[..., 0] = np.roll(band[..., 0], cs, axis=1)
            band[..., 2] = np.roll(band[..., 2], -cs, axis=1)
        if prev is not None and rng.random() < 0.35 * amt:
            band = np.roll(prev[y:y + h], shift // 2, axis=1)  # frame displacement
        out[y:y + h] = band
    if rng.random() < 0.6 * amt:  # macroblock corruption
        bw, bh = int(rng.uniform(0.2, 0.8) * W), int(rng.uniform(0.04, 0.25) * H)
        x0, y0 = int(rng.uniform(0, W - bw)), int(rng.uniform(0, H - bh))
        blk = max(8, int(W / 40))
        reg = out[y0:y0 + bh, x0:x0 + bw]
        small = cv2.resize(reg, (max(1, bw // blk), max(1, bh // blk)), interpolation=cv2.INTER_AREA)
        reg = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
        out[y0:y0 + bh, x0:x0 + bw] = reg
    if rng.random() < 0.3 * amt:  # horizontal tear line
        y = int(rng.uniform(0, H))
        out[y:y + 2] = np.clip(out[y:y + 2] * 2.5, 0, 1)
    return out


def bloom(img, amt):
    if amt < 0.01:
        return img
    H, W = img.shape[:2]
    y = img.mean(-1, keepdims=True)
    hi = np.clip((img - 0.6) * 2.5, 0, 1) * (y > 0.5)
    small = cv2.resize(hi, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    glow = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    x = img * (2 ** (amt * 1.6)) + glow * amt * 1.2
    knee = 0.8  # soft shoulder: blown highlights roll off instead of clipping flat
    return np.where(x < knee, x, knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee))))


@njit(parallel=True, cache=True, fastmath=True)
def _finish(img, vig, noise, dither, grain, out):
    H, W = img.shape[0], img.shape[1]
    for yy in prange(H):
        gy = yy * 0.5
        y0 = int(gy)
        fy = gy - y0
        for xx in range(W):
            gx = xx * 0.5
            x0 = int(gx)
            fx = gx - x0
            n = (noise[y0, x0] * (1 - fx) + noise[y0, x0 + 1] * fx) * (1 - fy) + \
                (noise[y0 + 1, x0] * (1 - fx) + noise[y0 + 1, x0 + 1] * fx) * fy
            v = vig[yy, xx, 0]
            r = img[yy, xx, 0] * v
            g = img[yy, xx, 1] * v
            b = img[yy, xx, 2] * v
            y = min(max((r + g + b) / 3.0, 0.0), 1.0)
            amp = grain * (0.35 + 0.65 * math.sqrt(min(y * (1.0 - y) * 4.0, 1.0)))
            dd = dither[yy, xx] - 0.5
            out[yy, xx, 0] = int(min(max((r + n * amp) * 255.0 + dd + 0.5, 0.0), 255.0))
            out[yy, xx, 1] = int(min(max((g + n * amp) * 255.0 + dd + 0.5, 0.0), 255.0))
            out[yy, xx, 2] = int(min(max((b + n * amp) * 255.0 + dd + 0.5, 0.0), 255.0))


# ---------------------------------------------------------------- renderer

class Renderer:
    def __init__(self, tl, preview=False):
        self.tl = tl
        self.preview = preview
        self.fps = tl.get("fps", 30)
        self.W, self.H = tl.get("width", 1080), tl.get("height", 1920)
        if preview:
            self.W, self.H = self.W // 2, self.H // 2
        self.clock = Clock(tl.get("beats"))
        self.fx = []
        for f in tl.get("fx", []):
            f = dict(f)
            f["_t"] = self.clock(f["t"])
            f["_d"] = self.clock(f["t_end"]) - f["_t"] if "t_end" in f else f.get("dur", 0.1)
            self.fx.append(f)
        self.shots = tl["shots"]
        t = 0.0
        for s in self.shots:  # chain shots: t0 defaults to previous t1
            s.setdefault("t0", t)
            t = s["t1"]
        self.duration = self.clock(tl.get("duration") or self.shots[-1]["t1"])
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        r = np.sqrt(((xx / self.W - 0.5) * 1.1) ** 2 + ((yy / self.H - 0.5) * 0.9) ** 2)
        self.vignette = (1 - np.clip((r - 0.35) * 0.9, 0, 0.45))[..., None].astype(np.float32)
        self.prev = None
        self.grain_rng = np.random.default_rng(7)

    def shot_at(self, t):
        for k, s in enumerate(self.shots):
            if self.clock(s["t0"]) <= t + 1e-6 < self.clock(s["t1"]):
                return k
        return len(self.shots) - 1

    def active(self, t):
        for f in self.fx:
            e, u = envelope(f, t, self.fps)
            if e > 0 or (0 <= u < 1):
                yield f, e, u

    def frame(self, shot, i, t, fi):
        W, H = self.W, self.H
        zoom, fx, fy, u_shot = shot.geometry(i)
        dx = dy = rot = 0.0
        dblur, dangle, rblur, rgb_px, rgb_rad = 0.0, 0.0, 0.0, 0.0, 0.0
        stops, flash, dip, bl, gl, ghost, inv, thr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        act = list(self.active(t))
        for f, e, u in act:
            a = f.get("amt", 1.0) * e
            typ = f["type"]
            if typ == "punch":
                zoom *= 1 + a * 0.25
            elif typ == "push":
                zoom *= 1 + f.get("amt", 0.1) * smooth(min(max(u, 0), 1))
            elif typ == "shake":
                seed = int(f["_t"] * 1000)
                tt = t * f.get("freq", 1.0)
                amp = a * 0.03
                dx += noise1(tt, seed) * amp * W
                dy += noise1(tt, seed + 1) * amp * H * 0.6
                rot += noise1(tt, seed + 2) * a * 1.8
                zoom *= 1 + amp * 2.2
                dblur = max(dblur, a * W * 0.02)
            elif typ in ("whip", "zoomtrans"):
                # centred on a cut: outgoing accelerates away, incoming settles in
                side = -1 if u < 0.5 else 1
                v = (u / 0.5) if u < 0.5 else ((1 - u) / 0.5)
                amt = f.get("amt", 1.0)
                if typ == "whip":
                    ang = f.get("angle", 0.0)
                    mag = v ** 3 * amt * 0.45 * W * (-side if u < 0.5 else side)
                    dx += mag * math.cos(math.radians(ang))
                    dy += mag * math.sin(math.radians(ang))
                    dblur = max(dblur, v ** 2 * amt * W * 0.18)
                    dangle = -ang
                else:
                    zoom *= 1 + v ** 2.5 * amt * 0.6
                    rblur = max(rblur, v ** 2 * amt * 0.35)
            elif typ == "dirblur":
                dblur = max(dblur, a * W * 0.12)
                dangle = f.get("angle", 0.0)
            elif typ == "zoomblur":
                rblur = max(rblur, a * 0.3)
            elif typ == "rgb":
                rgb_px += a * W * 0.012 * (1 if (fi // 2) % 2 == 0 or not f.get("jitter") else -1)
                rgb_rad += a * 0.006
            elif typ == "glitch":
                gl = max(gl, a)
            elif typ == "flash":
                flash = max(flash, a)
            elif typ == "dip":
                dip = max(dip, a)
            elif typ == "bloom":
                bl = max(bl, a)
            elif typ == "exposure":
                stops += a
            elif typ == "strobe":
                period = max(1, int(f.get("period", 2)))
                if (fi // period) % 2 == 1:
                    stops -= a * 3.5
                else:
                    stops += a * 0.4
            elif typ == "ghost":
                ghost = max(ghost, a * 0.6)
            elif typ == "invert":
                inv = max(inv, a)
            elif typ == "threshold":
                thr = max(thr, a)
        # ---- one resample: crop window + zoom + shake + rotation
        src = shot.sample(i)
        k = shot.k
        cw, ch = shot.cw * k / zoom, shot.ch * k / zoom
        cx = fx * shot.sw * k - shot.x0 * k
        cy = shot.sh * k * (0.5 + (fy - 0.5) * (1 - 1 / zoom))
        cx = float(np.clip(cx, cw / 2, src.shape[1] - cw / 2)) if src.shape[1] >= cw else src.shape[1] / 2
        cy = float(np.clip(cy, ch / 2, src.shape[0] - ch / 2))
        s = W / cw
        m = cv2.getRotationMatrix2D((cx, cy), rot, s)
        m[0, 2] += W / 2 - cx + dx
        m[1, 2] += H / 2 - cy + dy
        interp = cv2.INTER_CUBIC if s > 1.05 else cv2.INTER_AREA if s < 0.7 else cv2.INTER_LINEAR
        img = cv2.warpAffine(src.astype(np.float32), m, (W, H), flags=interp,
                             borderMode=cv2.BORDER_REFLECT)
        if s > 1.3:  # gentle restore of upscaled detail
            img = cv2.addWeighted(img, 1.35, cv2.GaussianBlur(img, (0, 0), 1.2), -0.35, 0)
        # ---- grade
        u = (t - self.clock(shot.s["t0"])) / max(1e-6, self.clock(shot.s["t1"]) - self.clock(shot.s["t0"]))
        img = grade(img, shot.s.get("grade", "base"),
                    kf(shot.s.get("exposure", 0.0), u) + stops)
        # ---- blur fx
        if dblur > 2:
            img = dir_blur(img, dblur, dangle)
        if rblur > 0.005:
            img = zoom_blur(img, rblur)
        if bl > 0:
            img = bloom(img, bl)
        if gl > 0:
            rng = np.random.default_rng(fi * 7919 + 13)
            img = glitch(img, gl, rng, self.prev)
            rgb_px += gl * W * 0.01 * (1 if fi % 2 else -1)
        if abs(rgb_px) > 0.5 or rgb_rad > 0.001:
            img = rgb_split(img, rgb_px, rgb_rad)
        if ghost > 0 and self.prev is not None:
            img = img * (1 - ghost) + np.maximum(img, self.prev) * ghost
        if thr > 0:
            y = img.mean(-1, keepdims=True)
            hard = np.clip((y - 0.45) * 8, 0, 1).repeat(3, -1)
            img = img * (1 - thr) + hard * thr
        if inv > 0:
            img = img * (1 - inv) + (1 - img) * inv
        if dip > 0:  # exposure crash that keeps the brightest highlights
            img = np.power(np.clip(img, 0, 1), 1 + dip * 7) * (1 - 0.6 * dip)
        if flash > 0:
            img = img + (1 - img) * flash
        # ---- finishing: vignette + luma-weighted grain + dither, straight to 8-bit
        self.prev = img
        n = self.grain_rng.standard_normal((H // 2 + 1, W // 2 + 1), dtype=np.float32)
        d = self.grain_rng.random((H, W), dtype=np.float32)
        out = np.empty((H, W, 3), np.uint8)
        _finish(np.ascontiguousarray(img, dtype=np.float32), self.vignette, n, d,
                float(self.tl.get("grain", 0.035)), out)
        return out

    def run(self, out, t_from=0.0, t_to=None):
        t_to = min(t_to or self.duration, self.duration)
        W, H, fps = self.W, self.H, self.fps
        f0, f1 = int(round(t_from * fps)), int(round(t_to * fps))
        tl = self.tl
        music = tl.get("music")
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
        if music:
            mstart = tl.get("music_start", 0.0) + t_from
            cmd += ["-ss", f"{mstart:.4f}", "-t", f"{t_to - t_from:.4f}", "-i", music]
        crf = "20" if self.preview else str(tl.get("crf", 15))
        cmd += ["-map", "0:v"] + (["-map", "1:a"] if music else [])
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                "-c:v", "libx264", "-preset", "veryfast" if self.preview else "slow",
                "-crf", crf, "-profile:v", "high", "-tune", "film",
                "-x264-params", "aq-mode=3:deblock=-1,-1",
                "-maxrate", tl.get("maxrate", "40M"), "-bufsize", tl.get("bufsize", "80M"),
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-color_range", "tv", "-g", str(fps * 2), "-movflags", "+faststart"]
        if music:
            fade = tl.get("fade_out", 0.12)
            cmd += ["-af", f"afade=t=out:st={max(0, t_to - t_from - fade):.3f}:d={fade}",
                    "-c:a", "aac", "-b:a", "320k", "-ar", "48000"]
        cmd += ["-shortest", out]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        cache = {}
        for fi in range(f0, f1):
            t = fi / fps
            k = self.shot_at(t)
            if k not in cache:
                cache.clear()
                spec = self.shots[k]
                spec["t0"] = self.clock(spec["t0"])
                cache[k] = Shot(spec, self.clock, W, H, fps, self.preview)
                sys.stderr.write(f"\r[{t:6.2f}s] shot {k + 1}/{len(self.shots)} "
                                 f"{os.path.basename(spec['src'])}          ")
            shot = cache[k]
            i = int(round(t * fps)) - int(round(self.clock(shot.s["t0"]) * fps))
            i = min(max(i, 0), shot.n - 1)
            img = self.frame(shot, i, t, fi)
            enc.stdin.write(img.tobytes())
        enc.stdin.close()
        enc.wait()
        sys.stderr.write("\n")
        if enc.returncode:
            raise SystemExit("encoder failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("out")
    ap.add_argument("--preview", action="store_true", help="half resolution, fast encode")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    a = ap.parse_args()
    with open(a.timeline) as fh:
        tl = json.load(fh)
    base = os.path.dirname(os.path.abspath(a.timeline))
    for key in ("beats", "music"):
        if tl.get(key) and not os.path.isabs(tl[key]):
            tl[key] = os.path.join(base, tl[key])
    for s in tl["shots"]:
        if not os.path.isabs(s["src"]):
            s["src"] = os.path.join(base, s["src"])
    Renderer(tl, a.preview).run(a.out, a.t_from, a.t_to)


if __name__ == "__main__":
    main()
