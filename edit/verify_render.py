#!/usr/bin/env python3
"""QC a rendered edit against its timeline: format, black frames, cut timing
vs the planned (beat-grid) cuts, audio loudness / true peak, and a contact
sheet with one frame from the middle of every shot.

Usage: verify_render.py timeline.json video.mp4 sheet.jpg
"""
import argparse
import json
import re
import subprocess

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("video")
    ap.add_argument("sheet")
    a = ap.parse_args()
    tl = json.load(open(a.timeline))
    fps = tl["fps"]
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels:"
         "format=duration,bit_rate", "-of", "json", a.video], capture_output=True, text=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    au = [s for s in info["streams"] if s["codec_type"] == "audio"]
    W, H = v["width"], v["height"]
    print(f"video {v['codec_name']} {W}x{H} ({W / H:.4f}; 16:9 = {16 / 9:.4f}, 9:16 = {9 / 16:.4f}) "
          f"{v['r_frame_rate']} "
          f"{v['pix_fmt']}, {float(info['format']['duration']):.2f}s, "
          f"{int(info['format']['bit_rate']) / 1e6:.1f} Mb/s")
    print("audio", [f"{s['codec_name']} {s['sample_rate']}Hz {s['channels']}ch" for s in au])

    # per-frame luma + difference at 1/4 res
    sw, sh = W // 4, H // 4
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.video, "-vf", f"scale={sw}:{sh}",
                          "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, sh, sw).astype(np.float32)
    mean = fr.mean((1, 2))
    n = len(fr)
    dark = mean < 10
    runs, i = [], 0
    while i < n:
        if dark[i]:
            j = i
            while j < n and dark[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    long_runs = [(s, e) for s, e in runs if e - s > 4]
    print(f"frames {n}; near-black (<10/255) frames {int(dark.sum())}; runs >4 frames: "
          + (", ".join(f"{s / fps:.2f}-{e / fps:.2f}s" for s, e in long_runs) or "none"))

    # planned cuts vs detected content changes (frame-to-frame difference peaks)
    diff = np.r_[0, np.abs(np.diff(fr, axis=0)).mean((1, 2))]
    planned = [s["t0"] for s in tl["shots"][1:]]
    offs, missed = [], []
    for t in planned:
        k = int(round(t * fps))
        lo, hi = max(1, k - 3), min(n - 1, k + 4)
        j = lo + int(np.argmax(diff[lo:hi]))
        if diff[j] < 4:
            missed.append(round(t, 2))
        else:
            offs.append(j - k)
    offs = np.array(offs)
    print(f"cuts planned {len(planned)}; detected {len(offs)}; frame offset from beat grid: "
          f"exact {np.mean(offs == 0) * 100:.0f}%, within 1 frame {np.mean(abs(offs) <= 1) * 100:.0f}%"
          + (f"; low-contrast cuts at {missed}" if missed else ""))

    # loudness
    log = subprocess.run(["ffmpeg", "-nostats", "-i", a.video, "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    summ = log[log.rfind("Summary:"):]
    I = re.search(r"I:\s+(-?[\d.]+) LUFS", summ)
    tp = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ)
    lra = re.search(r"LRA:\s+(-?[\d.]+) LU", summ)
    print(f"audio integrated {I.group(1)} LUFS, true peak {tp.group(1)} dBFS, LRA {lra.group(1)} LU")

    # contact sheet: middle of every shot + first/last frame
    cap = cv2.VideoCapture(a.video)
    times = [0.0] + [(s["t0"] + s["t1"]) / 2 for s in tl["shots"]] + [n / fps - 1 / fps]
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, img = cap.read()
        if not ok:
            continue
        tw = 108 if H > W else 192  # tile keeps the video's aspect
        img = cv2.resize(img, (tw, int(round(tw * H / W))), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{t:.1f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        tiles.append(img)
    cols = 16 if H > W else 10
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.sheet, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


if __name__ == "__main__":
    main()
