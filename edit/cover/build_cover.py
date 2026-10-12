#!/usr/bin/env python3
"""Build the TikTok cover: render edit/cover/cover.html (1080x1920) in headless
Chrome, then run cover_fx.py (bloom, glitch slices through the title, chromatic
aberration, vignette, grain) over it.

Usage: build_cover.py out.png [--chrome PATH] [--glitch 0.49,0.66]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHROMES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser",
]


def find_chrome(path):
    for c in ([path] if path else CHROMES):
        if c and (os.path.exists(c) or shutil.which(c)):
            return c
    raise SystemExit("no Chrome/Chromium found; pass --chrome")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--chrome")
    ap.add_argument("--glitch", default="0.49,0.66", help="title band (fractions of height)")
    a = ap.parse_args()
    chrome = find_chrome(a.chrome)
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base.png")
        page = "file:///" + os.path.join(HERE, "cover.html").replace("\\", "/")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", "--window-size=1080,1920",
                        "--virtual-time-budget=10000", f"--user-data-dir={os.path.join(tmp, 'profile')}",
                        f"--screenshot={base}", page], check=True, capture_output=True)
        subprocess.run([sys.executable, os.path.join(HERE, "cover_fx.py"), base, a.out,
                        "--glitch", a.glitch], check=True)
    print(a.out)


if __name__ == "__main__":
    main()
