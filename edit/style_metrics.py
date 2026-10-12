#!/usr/bin/env python3
"""Style metrics of an edit, comparable between videos of any size/orientation:
cut count and shot length, in-shot motion (optical flow, % of frame width per
frame), brightness, blown-white and dark frames, saturation, flicker density.

Reference edit (media/reference/ref.mov, 11 s, 16:9) measured with this script:
  motion median 0.26 %/frame, fast (>1 %) 18 %, static (<0.1 %) 27 %
  luma mean 91, frames >150: 12 %, frames <25: 2 %, chroma median 8 (3-14)
  flicker frames (luma diff > 40): 54 in 11 s (~4.9/s); ~21 scene cuts (~0.5 s/shot)
Without a timeline the cut detector also counts flashes/negatives as cuts (the
reference reads 38 cuts, 0.17 s median), so pass the timeline for your own edits.

Usage: style_metrics.py video.mp4 [timeline.json] [--per-shot]
"""
import json
import subprocess
import sys

import cv2
import numpy as np


def main():
    path = sys.argv[1]
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True).stdout)["streams"][0]
    n_, d_ = info["r_frame_rate"].split("/")
    fps = float(n_) / float(d_)
    W = 256 if info["width"] >= info["height"] else 144
    H = int(round(W * info["height"] / info["width"] / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}",
                          "-pix_fmt", "yuv444p", "-f", "rawvideo", "-"], capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, 3, H, W).astype(np.float32)
    Y = f[:, 0]
    n = len(f)
    luma = Y.mean((1, 2))
    chroma = np.sqrt((f[:, 1] - 128) ** 2 + (f[:, 2] - 128) ** 2).mean((1, 2))
    d = np.r_[0, np.abs(np.diff(Y, axis=0)).mean((1, 2))]
    if len(sys.argv) > 2 and sys.argv[2].endswith(".json"):
        tl = json.load(open(sys.argv[2]))
        cuts = [int(round(s["t0"] * fps)) for s in tl["shots"][1:]]
    else:
        hist = [cv2.calcHist([y.astype(np.uint8)], [0], None, [32], [0, 256]).ravel() / y.size
                for y in Y]
        hd = np.r_[0, [np.abs(hist[i] - hist[i - 1]).sum() for i in range(1, n)]]
        cuts = []
        for c in np.where((d > 18) & (hd > 0.25))[0]:
            if not cuts or c - cuts[-1] > 4:
                cuts.append(int(c))
    near = np.zeros(n, bool)
    for c in cuts:
        near[max(0, c - 3):c + 4] = True
    g = [cv2.GaussianBlur(y.astype(np.uint8), (0, 0), 1.2) for y in Y]
    mags, at = [], []
    for i in range(1, n):
        if near[i] or near[i - 1]:
            continue
        fl = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(np.median(np.linalg.norm(fl, axis=2)) / W * 100)
        at.append(i)
    mags = np.array(mags) if mags else np.zeros(1)
    if "--per-shot" in sys.argv:  # which shots read as static / fast
        bounds = [0] + cuts + [n]
        for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            m = np.array([v for v, i in zip(mags, at) if a <= i < b])
            if len(m):
                print(f"  shot {k:2d} {a / fps:6.2f}s  measured {len(m):3d}f  motion {np.median(m):.2f}  "
                      f"static {np.mean(m < 0.1) * 100:3.0f}%  fast {np.mean(m > 1) * 100:3.0f}%")
    lens = np.diff([0] + cuts + [n]) / fps
    dur = n / fps
    print(f"{path}: {info['width']}x{info['height']} {fps:.2f}fps {dur:.1f}s, "
          f"cuts {len(cuts)}, shot median {np.median(lens):.2f}s")
    print(f"  motion median {np.median(mags):.2f} %/frame, fast(>1%) {np.mean(mags > 1) * 100:.0f}%, "
          f"static(<0.1%) {np.mean(mags < 0.1) * 100:.0f}%")
    print(f"  luma mean {luma.mean():.0f}, >150 {np.mean(luma > 150) * 100:.0f}%, "
          f"<25 {np.mean(luma < 25) * 100:.0f}%, chroma median {np.median(chroma):.0f} "
          f"({np.percentile(chroma, 10):.0f}-{np.percentile(chroma, 90):.0f})")
    print(f"  flicker frames (diff>40) {int((d > 40).sum())} = {(d > 40).sum() / dur:.1f}/s")


if __name__ == "__main__":
    main()
