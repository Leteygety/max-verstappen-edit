#!/usr/bin/env python3
"""Check framing.json against the frame pool: every 9:16 window (same maths as
render.py, at zoom 1 / the frame's zoom) must stay clear of the burned-in
keep-out zones, with a margin for shake. Writes a contact sheet of the actual
crops for eyeballing.

Usage: check_framing.py edit/framing.json media/selected out_sheet.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np

MARGIN = 8  # source px: shake / rotation can pull this much extra into view


def key(filename):
    """abudhabi21__MTe12fH2xtQ_008.jpg -> abudhabi21_008"""
    stem = os.path.splitext(filename)[0]
    return stem.split("__")[0] + "_" + stem.rsplit("_", 1)[1]


def window(sw, sh, fx, zoom, fy, W=1080, H=1920):
    cw, ch = min(sw, sh * W / H), sh if sh * W / H < sw else sw * H / W
    w, h = cw / zoom, ch / zoom
    cx = float(np.clip(fx * sw, w / 2, sw - w / 2))
    cy = float(np.clip(sh * (0.5 + (fy - 0.5) * (1 - 1 / zoom)), h / 2, sh - h / 2))
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def hits(win, box, m=MARGIN):
    x0, y0, x1, y1 = win
    return not (x1 + m <= box[0] or x0 - m >= box[2] or y1 + m <= box[1] or y0 - m >= box[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("framing")
    ap.add_argument("frames")
    ap.add_argument("sheet")
    a = ap.parse_args()
    cfg = json.load(open(a.framing))
    # frames are keyed "<source>_<index>" (the video id in the file name is dropped)
    files = {key(f): f for f in os.listdir(a.frames) if f.endswith(".jpg")}
    names = sorted(files)
    missing = sorted(set(names) - set(cfg["frames"]))
    unknown = sorted(set(cfg["frames"]) - set(names))
    bad, tiles = [], []
    for n in names:
        if n not in cfg["frames"]:
            continue
        f = cfg["frames"][n]
        img = cv2.imread(os.path.join(a.frames, files[n]))
        sh, sw = img.shape[:2]
        boxes = cfg["keepout_all"] + f.get("keepout", [])
        if f.get("skip"):
            continue
        win = window(sw, sh, f["fx"], f.get("zoom", 1.0), f.get("fy", 0.5))
        clash = [b for b in boxes if hits(win, b)]
        if clash:
            bad.append((n, [round(v) for v in win], clash))
        x0, y0, x1, y1 = (int(round(v)) for v in win)
        crop = cv2.resize(img[y0:y1, x0:x1], (135, 240), interpolation=cv2.INTER_AREA)
        if clash:
            cv2.rectangle(crop, (0, 0), (134, 239), (0, 0, 255), 4)
        cv2.putText(crop, n[:9] + n[-3:], (2, 234), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, (0, 255, 255), 1)
        tiles.append(crop)
    cols = 13
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.sheet, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
    skipped = [n for n, f in cfg["frames"].items() if f.get("skip")]
    print(f"{len(names)} frames, {len(names) - len(missing) - len(skipped)} framed, "
          f"{len(skipped)} skipped, {len(missing)} missing, {len(unknown)} unknown names")
    for n in missing + unknown:
        print("  name mismatch:", n)
    for n, w, c in bad:
        print(f"  KEEP-OUT HIT {n}: window {w} vs {c}")
    if not (missing or unknown or bad):
        print("all windows clear of burned-in text")


if __name__ == "__main__":
    main()
