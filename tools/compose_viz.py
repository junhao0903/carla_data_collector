#!/usr/bin/env python3
"""Compose visualization outputs into a single composite image grid.

Layout:
  Row 1: semantic_viz | annotations_viz | depth_viz
  Row 2: occ_viz      | trajectory_viz   | projection

All 6 images resized to uniform cell size.

Usage:
    python tools/compose_viz.py output/<run>                    # → PNG frames
    python tools/compose_viz.py output/<run> --format gif       # → composite.gif
    python tools/compose_viz.py output/<run> --format video     # → composite.mp4
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def _load_image(base_dirs, subdir, frame):
    for base in base_dirs:
        for ext in (".png", ".jpg"):
            p = os.path.join(base, subdir, f"{frame:08d}{ext}")
            if os.path.exists(p):
                return Image.open(p)
    return None


def _resize_to(img, target_w, target_h):
    """Resize to target, keep aspect ratio, pad black bars."""
    iw, ih = img.size
    scale = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(img, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def compose_run(run_dir, output_format="img", fps=10, channel="CAM_FRONT",
                cell_w=640, cell_h=480):
    """Generate composite frames (img) or GIF/video."""
    occ_dir = os.path.join(run_dir, "OCC", "occ_viz")
    if not os.path.isdir(occ_dir):
        print("No OCC/occ_viz/ found, run npy2jpg first")
        return

    frames = sorted([int(f.split(".")[0])
                     for f in os.listdir(occ_dir)
                     if f.endswith((".png", ".jpg"))])
    if not frames:
        print("No frames found in OCC/occ_viz/")
        return

    cam_base = os.path.join(run_dir, channel)
    occ_base = os.path.join(run_dir, "OCC")
    traj_base = os.path.join(run_dir, "TRAJ")

    has_projection = os.path.isdir(os.path.join(occ_base, "projection"))

    out_dir = os.path.join(run_dir, "COMPOSITE")
    os.makedirs(out_dir, exist_ok=True)

    # Try loading a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    print(f"Composing {len(frames)} frames → {out_dir}/")

    for frame in tqdm(frames, desc="Composing"):
        # Skip frames where any expected source is missing
        sem = _load_image([cam_base], "semantic_viz", frame)
        ann = _load_image([cam_base], "annotations_viz", frame)
        dep = _load_image([cam_base], "depth_viz", frame)
        occ = _load_image([occ_base], "occ_viz", frame)
        traj = _load_image([traj_base], "trajectory_viz", frame)
        proj = _load_image([occ_base], "projection", frame) if has_projection else None

        if not all([sem, ann, dep, occ]):
            continue

        cols = 3
        rows = 2
        canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), (0, 0, 0))

        placements = [
            (sem, 0, 0, "semantic_viz"),
            (ann, 1, 0, "annotations_viz"),
            (dep, 2, 0, "depth_viz"),
            (occ, 0, 1, "occ_viz"),
            (traj, 1, 1, "trajectory_viz"),
            (proj, 2, 1, "projection"),
        ]

        for img, col, row, name in placements:
            if img is None:
                continue
            thumb = _resize_to(img, cell_w, cell_h)
            x, y = col * cell_w, row * cell_h
            canvas.paste(thumb, (x, y))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([x, y, x + len(name) * 8, y + 18], fill=(0, 0, 0, 160))
            draw.text((x + 2, y + 2), name, fill=(255, 255, 255), font=font)

        canvas.save(os.path.join(out_dir, f"{frame:08d}.png"))

    # Use only actually saved frames for GIF/MP4
    saved_frames = sorted([int(f.split(".")[0]) for f in os.listdir(out_dir)
                           if f.endswith(".png")])
    if output_format == "gif":
        _make_gif(out_dir, saved_frames, fps, cleanup=True)
    elif output_format == "video":
        _make_mp4(out_dir, saved_frames, fps, cleanup=True)


def _make_gif(out_dir, frames, fps, cleanup=False):
    gif_path = os.path.join(out_dir, "composite.gif")
    images = []
    for frame in tqdm(frames, desc="Building GIF"):
        img = Image.open(os.path.join(out_dir, f"{frame:08d}.png"))
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        img = img.quantize(colors=256, method=Image.MEDIANCUT)
        images.append(img)
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"GIF saved: {gif_path}")
    if cleanup:
        for frame in frames:
            os.remove(os.path.join(out_dir, f"{frame:08d}.png"))


def _make_mp4(out_dir, frames, fps, cleanup=False):
    import subprocess
    mp4_path = os.path.join(out_dir, "composite.mp4")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(out_dir, "%08d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        mp4_path,
    ]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(mp4_path):
        print(f"MP4 saved: {mp4_path}")
        if cleanup:
            for frame in frames:
                os.remove(os.path.join(out_dir, f"{frame:08d}.png"))
    else:
        print("MP4 generation failed (ffmpeg not found?)")


def main():
    parser = argparse.ArgumentParser(
        description="Compose visualization outputs into a single grid image/GIF/MP4")
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--format", choices=["img", "gif", "video"], default="img",
                        help="Output format: img (PNG frames), gif, video (MP4)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--channel", default="CAM_FRONT")
    parser.add_argument("--cell-w", type=int, default=480)
    parser.add_argument("--cell-h", type=int, default=360)
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        dirs = sorted(glob.glob("output/*"))
        if not dirs:
            print("No output directories found")
            return
        run_dir = dirs[-1]
        print(f"Using latest: {run_dir}")

    compose_run(run_dir, args.format, args.fps, args.channel, args.cell_w, args.cell_h)


if __name__ == "__main__":
    main()
