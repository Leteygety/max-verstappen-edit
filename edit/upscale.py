#!/usr/bin/env python3
"""4x super-resolution of still frames with Real-ESRGAN's compact model
(realesr-general-x4v3, SRVGGNetCompact), CPU only. Reads the source folder,
writes PNGs with the same stem into the output folder; sources are untouched.

Usage: upscale.py media/selected media/work/sr/x4 --weights realesr-general-x4v3.pth
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body += [nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1)]
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--only", nargs="*", help="process only stems containing these strings")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    net = SRVGGNetCompact()
    sd = torch.load(a.weights, map_location="cpu")
    net.load_state_dict(sd.get("params", sd))
    net.eval()
    for path in sorted(glob.glob(os.path.join(a.src, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if a.only and not any(s in stem for s in a.only):
            continue
        dst = os.path.join(a.out, stem + ".png")
        if os.path.exists(dst):
            continue
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        with torch.inference_mode():
            y = net(torch.from_numpy(img).permute(2, 0, 1)[None])[0]
        y = (y.clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(dst, cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
        print(stem, y.shape[1], "x", y.shape[0], flush=True)


if __name__ == "__main__":
    main()
