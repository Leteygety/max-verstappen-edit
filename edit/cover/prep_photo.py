#!/usr/bin/env python3
"""Turn a trackside car photo into the cover's hero layer: GrabCut the car,
pan-blur and darken the track around it, grade both like the edit.

Usage: prep_photo.py photo out.png --rect x,y,w,h --crop x,y,w,h
                     [--width 1080] [--angle 116] [--blur 70]
  --rect   box around the car (source pixels) for GrabCut
  --crop   region of the photo kept (keep watermarks outside it)
  --angle  direction of travel in image degrees (0 = right, 90 = down)
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from render import grade  # noqa: E402


def box(s):
    return [int(v) for v in s.split(",")]


def line_blur(img, length, angle):
    n = max(3, int(length) | 1)
    k = np.zeros((n, n), np.float32)
    c = n // 2
    dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    for t in np.linspace(-c, c, n * 2):
        k[int(round(c + t * dy)), int(round(c + t * dx))] = 1
    return cv2.filter2D(img, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out")
    ap.add_argument("--rect", type=box, required=True)
    ap.add_argument("--crop", type=box, required=True)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--angle", type=float, default=116)
    ap.add_argument("--blur", type=float, default=70)
    a = ap.parse_args()

    rgb = np.array(Image.open(a.photo).convert("RGB"))
    mask = np.zeros(rgb.shape[:2], np.uint8)
    bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, tuple(a.rect), bgm, fgm, 6,
                cv2.GC_INIT_WITH_RECT)
    car = np.where((mask == 1) | (mask == 3), 1.0, 0.0).astype(np.float32)
    # GrabCut keeps asphalt seen through the floor/wings: drop mid-grey, unsaturated pixels.
    f = rgb.astype(np.float32) / 255
    luma = f @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sat = f.max(-1) - f.min(-1)
    asphalt = ((luma > 0.28) & (luma < 0.7) & (sat < 0.1)).astype(np.uint8)
    asphalt = cv2.morphologyEx(asphalt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    car[asphalt > 0] = 0
    car = cv2.morphologyEx(car, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    x, y, w, h = a.crop
    s = a.width / w
    size = (a.width, int(round(h * s)))
    img = cv2.resize(rgb[y:y + h, x:x + w], size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255
    car = cv2.resize(car[y:y + h, x:x + w], size, interpolation=cv2.INTER_AREA)
    car = cv2.GaussianBlur(car, (0, 0), 1.5)[..., None]

    # Track: fill the car hole before blurring so the car doesn't smear into it.
    hole = (car[..., 0] > 0.02).astype(np.uint8)
    track = cv2.inpaint((img * 255).astype(np.uint8), cv2.dilate(hole, np.ones((9, 9))), 9,
                        cv2.INPAINT_TELEA).astype(np.float32) / 255
    track = line_blur(track, a.blur, a.angle)
    track = grade(np.ascontiguousarray(track), "cold", -1.5)
    body = grade(np.ascontiguousarray(img), "base", -0.25)
    out = body * car + track * (1 - car)
    Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(a.out)
    print(a.out, size)


if __name__ == "__main__":
    main()
