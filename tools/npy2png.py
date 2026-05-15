#!/usr/bin/env python3
"""Convert RGB .npy files to .png in a run directory."""
import sys
import os
import glob
import numpy as np
from PIL import Image


def convert_run(run_dir):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        channel = os.path.basename(cam_dir)
        npy_files = glob.glob(os.path.join(cam_dir, "*.npy"))
        if not npy_files:
            continue
        print(f"{channel}: {len(npy_files)} files")
        for npy_path in npy_files:
            png_path = npy_path.replace(".npy", ".png")
            if os.path.exists(png_path):
                continue
            arr = np.load(npy_path)
            arr = arr[:, :, :3]  # RGBA → RGB
            Image.fromarray(arr).save(png_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/npy2png.py output/<run_dir>")
        sys.exit(1)
    convert_run(sys.argv[1])
