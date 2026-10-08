#!/usr/bin/env python3
"""Finish the rendered cover layout with the edit's look: highlight bloom,
glitch slices through the title, radial chromatic aberration, vignette, grain.

Usage: cover_fx.py base.png out.png [--seed N]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from render import _finish, bloom, rgb_split  # noqa: E402


def slice_glitch(img, rng, y0, y1, n):
    H, W = img.shape[:2]
    out = img.copy()
    for _ in range(n):
        h = int(rng.uniform(0.003, 0.009) * H)
        y = int(rng.uniform(y0, y1 - h))
        shift = int(rng.choice([-1, 1]) * rng.uniform(0.012, 0.035) * W)
        band = np.roll(out[y:y + h], shift, axis=1)
        cs = int(rng.uniform(3, 8))
        band[..., 0] = np.roll(band[..., 0], cs, axis=1)
        band[..., 2] = np.roll(band[..., 2], -cs, axis=1)
        out[y:y + h] = band
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=33)
    a = ap.parse_args()
    img = cv2.cvtColor(cv2.imread(a.base), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    H, W = img.shape[:2]
    rng = np.random.default_rng(a.seed)
    img = slice_glitch(img, rng, int(H * 0.28), int(H * 0.43), 3)
    img = bloom(img, 0.18)
    img = rgb_split(img, 0.0, 0.0018)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx / W - 0.5) * 1.1) ** 2 + ((yy / H - 0.47) * 0.85) ** 2)
    vig = (1 - np.clip((r - 0.3) * 1.1, 0, 0.6))[..., None].astype(np.float32)
    noise = rng.standard_normal((H // 2 + 1, W // 2 + 1), dtype=np.float32)
    dither = rng.random((H, W), dtype=np.float32)
    out = np.empty((H, W, 3), np.uint8)
    _finish(np.ascontiguousarray(np.clip(img, 0, 1)), vig, noise, dither, 0.035, out)
    cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])


if __name__ == "__main__":
    main()
