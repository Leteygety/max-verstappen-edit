#!/usr/bin/env python3
"""Title layers for the v2 silence: the word VERSTAPPEN and Max's MV logo, as
tight RGBA PNGs (white, soft glow, 2x master resolution), plus a soft dark scrim
to sit under them, for render.py's "overlay" fx, which fades / blurs / scales
them over the footage.

The MV logo is redrawn as vectors from the reference artwork the user supplied
(an outlined V with swept wings and two outlined triangles beside its arms; the
registered mark is left out): outer contours below, inner contours are the
outer ones offset inward by the stroke width with mitred corners.

Usage: make_title.py out_dir [--font C:/Windows/Fonts/bahnschrift.ttf]
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WHITE = (246, 246, 244)


# MV logo in the artwork's own units (750 px canvas, symmetric about x = 368)
MV_V = [(120, 243), (277, 268), (368, 377), (459, 268), (616, 243), (368, 490)]
MV_TRI_L = [(120, 441), (203, 361), (283, 441)]
MV_TRI_R = [(453, 441), (533, 361), (616, 441)]
MV_STROKE = 13


def inset(poly, d):
    """Polygon offset inward by d with mitred corners (works for reflex corners too)."""
    p = np.array(poly, np.float64)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    sgn = 1.0 if area > 0 else -1.0  # inward normal side depends on orientation
    lines = []
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        t = (b - a) / np.linalg.norm(b - a)
        n = sgn * np.array([-t[1], t[0]])
        lines.append((a + n * d, t))
    out = []
    for i in range(len(p)):
        (p1, t1), (p2, t2) = lines[i - 1], lines[i]
        m = np.array([t1, -t2]).T
        s_, _ = np.linalg.solve(m, p2 - p1)
        out.append(tuple(p1 + t1 * s_))
    return out


def mv_alpha(scale):
    """The outlined MV mark rasterised at `scale` px per artwork unit (4x supersampled)."""
    ss = 4
    k = scale * ss
    x0, y0, x1, y1 = 120, 243, 616, 490
    pad = 20
    w, h = int((x1 - x0 + 2 * pad) * k), int((y1 - y0 + 2 * pad) * k)
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    tr = lambda poly: [((x - x0 + pad) * k, (y - y0 + pad) * k) for x, y in poly]  # noqa: E731
    for poly in (MV_V, MV_TRI_L, MV_TRI_R):
        d.polygon(tr(poly), fill=255)
        d.polygon(tr(inset(poly, MV_STROKE)), fill=0)
    img = img.resize((w // ss, h // ss), Image.LANCZOS)
    return np.asarray(img, np.float32) / 255


def with_glow(alpha, sigma, strength):
    glow = cv2.GaussianBlur(alpha, (0, 0), sigma)
    a = np.clip(np.maximum(alpha, glow * strength), 0, 1)
    rgba = np.zeros(alpha.shape + (4,), np.uint8)
    rgba[..., :3] = WHITE
    rgba[..., 3] = (a * 255).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def text_alpha(word, font_path, size, tracking, variation):
    font = ImageFont.truetype(font_path, size)
    font.set_variation_by_name(variation)
    adv = [font.getlength(c) for c in word]
    width = int(sum(adv) + tracking * size * (len(word) - 1)) + size
    asc, desc = font.getmetrics()
    img = Image.new("L", (width, asc + desc + size // 2), 0)
    d = ImageDraw.Draw(img)
    x = size / 2
    for c, a in zip(word, adv):
        d.text((x, size // 4), c, font=font, fill=255)
        x += a + tracking * size
    al = np.asarray(img, np.float32) / 255
    ys, xs = np.nonzero(al > 0.02)
    pad = size // 3
    return al[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--font", default="C:/Windows/Fonts/bahnschrift.ttf")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # 2x masters: the renderer only ever scales them down
    word = text_alpha("VERSTAPPEN", a.font, 240, 0.34, "SemiBold")
    with_glow(word, 14, 0.35).save(os.path.join(a.out, "verstappen_text.png"))
    logo = mv_alpha(2.0)
    with_glow(logo, 14, 0.35).save(os.path.join(a.out, "verstappen_mv.png"))
    # soft dark scrim laid under the title so it reads over a busy frame
    yy, xx = np.mgrid[-1:1:600j, -1:1:1200j]
    sc = np.clip(1 - np.hypot(xx, yy * 1.1), 0, 1) ** 1.6
    rgba = np.zeros(sc.shape + (4,), np.uint8)
    rgba[..., 3] = (sc * 255).round().astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(os.path.join(a.out, "verstappen_scrim.png"))
    print(f"text {word.shape[1]}x{word.shape[0]}, logo {logo.shape[1]}x{logo.shape[0]} -> {a.out}")


if __name__ == "__main__":
    main()
