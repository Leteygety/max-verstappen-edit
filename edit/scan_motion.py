#!/usr/bin/env python3
"""Motion map of a source clip for picking video in-points: scene cuts (ffmpeg
scene score at full frame rate), and per source shot the in-frame motion
(median optical flow, % of frame width per 30 fps output frame, the same unit
style_metrics.py reports) and brightness.

Usage: scan_motion.py clip.mkv [--from S] [--to S] [--json out.json]
"""
import argparse
import json
import subprocess

import cv2
import numpy as np

RATE = 10.0  # analysis frames per second
W = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--json")
    a = ap.parse_args()
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", a.clip], capture_output=True, text=True).stdout)
    s = info["streams"][0]
    H = int(round(W * s["height"] / s["width"] / 2) * 2)
    seek = ["-ss", str(a.t0)] + (["-t", str(a.t1 - a.t0)] if a.t1 else [])
    raw = subprocess.run(["ffmpeg", "-v", "error", *seek, "-i", a.clip, "-an", "-vf",
                          f"fps={RATE},scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    Y = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    n = len(Y)
    t = a.t0 + np.arange(n) / RATE
    # scene cuts at full frame rate (ffmpeg scene score); the 10 fps analysis frames
    # are only used for motion, so fast pans and blur do not read as cuts
    log = subprocess.run(["ffmpeg", "-v", "info", *seek, "-i", a.clip, "-an", "-vf",
                          "scale=320:-2,select='gt(scene,0.2)',showinfo", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    cut_t = [a.t0 + float(x.split("pts_time:")[1].split()[0])
             for x in log.splitlines() if "pts_time:" in x]
    cut = np.zeros(n, bool)
    for c in cut_t:
        cut[min(n - 1, int(np.ceil((c - a.t0) * RATE - 1e-6)))] = True
    mot = np.zeros(n)
    g = [cv2.GaussianBlur(y, (0, 0), 1.2) for y in Y]
    for i in range(1, n):
        if cut[i]:
            continue
        fl = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mot[i] = np.median(np.linalg.norm(fl, axis=2)) / W * 100 * RATE / 30.0
    edges = [a.t0] + [c for c in cut_t if c > a.t0 + 0.05] + [a.t0 + n / RATE]
    shots = []
    for s0, s1 in zip(edges[:-1], edges[1:]):
        if s1 - s0 < 0.15:
            continue
        idx = np.nonzero((t > s0 + 1e-6) & (t < s1 - 1e-6) & ~cut)[0]
        idx = idx[idx > 0]
        m = mot[idx] if len(idx) else np.zeros(1)
        i0, i1 = int((s0 - a.t0) * RATE), max(int((s1 - a.t0) * RATE), int((s0 - a.t0) * RATE) + 1)
        shots.append(dict(start=round(float(s0), 3), end=round(float(s1), 3),
                          motion=round(float(np.median(m)), 2), motion_max=round(float(m.max()), 2),
                          fast=round(float(np.mean(m > 1)), 2), luma=int(Y[i0:i1].mean()),
                          prof=[round(float(x), 1) for x in m[::2]]))
    for sh in shots:
        print(f"{sh['start']:7.2f}-{sh['end']:7.2f} ({sh['end'] - sh['start']:4.1f}s) "
              f"motion {sh['motion']:4.2f} max {sh['motion_max']:4.2f} fast {sh['fast']:.2f} "
              f"luma {sh['luma']:3d} | {' '.join(f'{x:.1f}' for x in sh['prof'][:24])}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(clip=a.clip, shots=shots), fh, indent=1)


if __name__ == "__main__":
    main()
