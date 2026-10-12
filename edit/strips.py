#!/usr/bin/env python3
"""Contact strips of candidate source moments for shot picking: one row per
segment (clip, start, end), N evenly spaced frames, each tile timecoded with
its source time.

Usage: strips.py out.jpg clip:start:end [clip:start:end ...] [--n 8] [--w 320]
"""
import argparse
import subprocess

import cv2
import numpy as np


def grab(path, t, w):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                          "-vf", f"scale={w}:-2", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    h = len(raw) // (w * 3)
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy() if h else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("segs", nargs="+")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--w", type=int, default=320)
    a = ap.parse_args()
    rows = []
    for seg in a.segs:
        path, t0, t1 = seg.rsplit(":", 2)
        t0, t1 = float(t0), float(t1)
        tiles = []
        for t in np.linspace(t0, t1, a.n):
            img = grab(path, t, a.w)
            if img is None:
                img = np.zeros((a.w * 9 // 16, a.w, 3), np.uint8)
            cv2.putText(img, f"{t:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, f"{t:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(img)
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 2, cv2.BORDER_CONSTANT) for t in tiles]
        row = np.hstack(tiles)
        label = path.replace("\\", "/").split("/")[-1].split("__")[0]
        cv2.putText(row, label, (a.w - 120, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rows.append(cv2.copyMakeBorder(row, 0, 2, 0, 0, cv2.BORDER_CONSTANT))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
