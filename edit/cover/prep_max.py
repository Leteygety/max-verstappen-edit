#!/usr/bin/env python3
"""Background layer for the cover: Max showing four fingers for his fourth title
(Las Vegas 2024), taken from his own world-champion video. Local contrast keeps
the black glove apart from the visor and the blue chin guard, the edit's grade
keeps the Red Bull colours, and a pool of light on Max lets the edges fall into
the dark. The crop is 1:1 source pixels (no upscale), clear of the F1 bug.

Usage: prep_max.py out.png [--clip CLIP] [--t 5.78] [--x 420] [--w 1080]
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from render import grade  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--clip", default="media/clips/max_helmet__RkpQvhg-IWw.mkv")
    ap.add_argument("--t", type=float, default=5.78, help="sharpest four-finger frame")
    ap.add_argument("--x", type=int, default=420, help="left edge of the crop (source px)")
    ap.add_argument("--w", type=int, default=1080, help="crop width = cover width")
    ap.add_argument("--exposure", type=float, default=0.1)
    ap.add_argument("--grade", default="base")
    ap.add_argument("--clahe", type=float, default=2.2)
    ap.add_argument("--hand", type=lambda v: [float(x) for x in v.split(",")], default=[420, 430, 230],
                    help="raised hand centre x, y and radius in crop pixels")
    ap.add_argument("--clarity", type=float, default=1.2)
    ap.add_argument("--spot", type=lambda v: [float(x) for x in v.split(",")], default=[0.45, 0.42],
                    help="centre of the light pool (fractions of the crop)")
    a = ap.parse_args()
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a.t:.3f}", "-i", a.clip, "-frames:v", "1",
                          "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)[:, a.x:a.x + a.w].astype(np.float32) / 255
    # local contrast on lightness so the black glove separates from the visor and the
    # blue chin guard behind it, then a gentle detail restore of the 1080p50 web source
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l8 = (lab[..., 0] * 2.55).astype(np.uint8)
    l8 = cv2.createCLAHE(clipLimit=a.clahe, tileGridSize=(8, 8)).apply(l8)
    lab[..., 0] = l8.astype(np.float32) / 2.55
    img = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    img = cv2.addWeighted(img, 1.3, cv2.GaussianBlur(img, (0, 0), 1.1), -0.3, 0)
    # extra clarity on the raised hand so the four gloved fingers read one by one
    hx, hy, hr = a.hand
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    hm = np.clip(1.4 - np.hypot(xx - hx, yy - hy) / hr, 0, 1)[..., None]
    detail = img - cv2.GaussianBlur(img, (0, 0), 5)
    img = np.clip(img + detail * a.clarity * hm, 0, 1)
    # the edit's base grade keeps Red Bull blue / red; light falls off away from Max
    img = np.clip(grade(img, a.grade, a.exposure), 0, 1)
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = a.spot[0] * W, a.spot[1] * H
    r = np.hypot((xx - cx) / (0.5 * W), (yy - cy) / (0.6 * H))
    spot = 0.22 + 0.78 * np.clip(1.2 - r, 0, 1) ** 1.4
    img = img * spot[..., None]
    out = (img * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"{a.out}: {a.w}x1080 from {os.path.basename(a.clip)} @ {a.t:.2f}s, x {a.x}-{a.x + a.w}")


if __name__ == "__main__":
    main()
