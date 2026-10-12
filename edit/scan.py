#!/usr/bin/env python3
"""Index source clips for shot picking: resolution/fps, scene cuts, and a
timecoded contact sheet per clip (one tile every --every seconds).

Usage: scan.py media/clips out_dir [--every 2]
"""
import argparse
import glob
import json
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--every", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = {}
    for path in sorted(glob.glob(os.path.join(a.clips, "*"))):
        name = os.path.splitext(os.path.basename(path))[0]
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
            capture_output=True, text=True).stdout)
        st, dur = info["streams"][0], float(info["format"]["duration"])
        log = subprocess.run(
            ["ffmpeg", "-v", "info", "-i", path, "-an", "-vf",
             "scale=320:-2,select='gt(scene,0.3)',showinfo", "-f", "null", "-"],
            capture_output=True, text=True).stderr
        cuts = [round(float(x.split("pts_time:")[1].split()[0]), 2)
                for x in log.splitlines() if "pts_time:" in x]
        n = max(1, int(dur / a.every))
        cols = 8
        rows = (n + cols - 1) // cols
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", path, "-an", "-vf",
             f"fps=1/{a.every},scale=320:-2,drawtext=text='%{{pts\\:hms}}':x=4:y=4:fontsize=18:"
             f"fontcolor=yellow:box=1:boxcolor=black@0.6,tile={cols}x{rows}",
             "-frames:v", "1", os.path.join(a.out, f"{name}.jpg")])
        index[name] = {"path": path, "w": st["width"], "h": st["height"],
                       "fps": st["r_frame_rate"], "duration": dur, "cuts": cuts}
        print(f"{name}: {st['width']}x{st['height']} {st['r_frame_rate']} {dur:.0f}s "
              f"{len(cuts)} cuts")
    with open(os.path.join(a.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)


if __name__ == "__main__":
    main()
