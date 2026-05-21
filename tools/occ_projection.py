#!/usr/bin/env python3
"""Project OCC voxels onto camera image for validation.

Usage:
    python tools/occ_projection.py output/<run> [--frame N]
"""
import argparse
import csv
import json
import math as m
import os
import sys

import numpy as np
from PIL import Image

# OCC label → color (matching README)
LABEL_COLORS = {
    0: (0, 0, 0),
    1: (128, 64, 128),    # road
    2: (244, 35, 232),    # sidewalk
    3: (70, 70, 70),      # building
    4: (102, 102, 156),   # wall
    5: (190, 153, 153),   # fence
    6: (153, 153, 153),   # pole
    8: (220, 220, 0),     # traffic_sign
    9: (107, 142, 35),    # vegetation
    12: (220, 20, 60),    # pedestrian
    14: (0, 0, 142),      # car
    15: (0, 0, 70),       # truck
    16: (0, 60, 100),     # bus
    18: (0, 0, 230),      # motorcycle
    19: (119, 11, 32),    # bicycle
}


def load_ego_poses(ego_csv):
    poses = {}
    with open(ego_csv) as f:
        for row in csv.DictReader(f):
            # Ego world pose (left-hand convention in CSV → CARLA world)
            frame = int(row["frame"])
            ex = float(row["x"])
            ey = -float(row["y_left"])  # left→right
            ez = float(row["z"])
            er = -float(row["roll_left"])
            ep = float(row["pitch"])
            eyaw = -float(row["yaw_left"])
            poses[frame] = (ex, ey, ez, er, ep, eyaw)
            # Camera transform in CARLA world
            if "cam_x" in row:
                cx = float(row["cam_x"])
                cy = -float(row["cam_y_left"])
                cz = float(row["cam_z"])
                cr = -float(row["cam_roll_left"])
                cp = float(row["cam_pitch"])
                cyaw = -float(row["cam_yaw_left"])
                poses[frame] = (*poses[frame], cx, cy, cz, cr, cp, cyaw)
    return poses


def load_camera_intrinsics(run_dir, channel):
    # Try loading from sensor_layout.yaml
    layout_path = os.path.join(run_dir, "sensor_layout.yaml")
    if os.path.exists(layout_path):
        import yaml
        with open(layout_path) as f:
            layout = yaml.safe_load(f)
        for s in layout.get("sensors", []):
            if s.get("channel") == channel and s.get("modality") == "camera_rgb":
                out = s.get("output", {})
                w = out.get("width", 1600)
                h = out.get("height", 900)
                hfov = m.radians(out.get("fov", 70))
                vfov = 2 * m.atan(m.tan(hfov / 2) * h / w)
                fx = w / (2 * m.tan(hfov / 2))
                fy = h / (2 * m.tan(vfov / 2))
                return {"fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2, "width": w, "height": h}
    # Fallback
    return {"fx": 1600, "fy": 1600, "cx": 800, "cy": 450, "width": 1600, "height": 900}


def run(run_dir, frame_number=None, sample=1, channel="CAM_FRONT"):
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if not os.path.exists(ego_csv):
        print("No ego_trajectory.csv found")
        return

    poses = load_ego_poses(ego_csv)
    K = load_camera_intrinsics(run_dir, channel)
    w, h = K["width"], K["height"]

    occ_npy_dir = os.path.join(run_dir, "OCC", "original")
    cam_img_dir = os.path.join(run_dir, channel, "original")
    out_dir = os.path.join(run_dir, "OCC", f"projection_{channel}")
    os.makedirs(out_dir, exist_ok=True)

    occ_files = sorted(os.listdir(occ_npy_dir)) if os.path.isdir(occ_npy_dir) else []
    if not occ_files:
        print("No OCC npy files found")
        return

    # Load camera mount transform from sensor layout
    cam_mount = None
    layout_path = os.path.join(run_dir, "sensor_layout.yaml")
    if os.path.exists(layout_path):
        import yaml
        with open(layout_path) as f:
            layout = yaml.safe_load(f)
        for s in layout.get("sensors", []):
            if s.get("channel") == channel and s.get("modality") == "camera_rgb":
                t = s.get("transform", {})
                cam_mount = (t.get("x", 1.5), t.get("y", 0.0), t.get("z", 1.6),
                            t.get("roll", 0.0), t.get("pitch", 0.0), t.get("yaw", 0.0))
                break
    if cam_mount is None:
        cam_mount = (1.5, 0.0, 1.6, 0.0, 0.0, 0.0)
    mx, my, mz, mroll, mpitch, myaw_m = cam_mount

    # Build camera world → image projection matrix
    # Camera intrinsic matrix
    Kmat = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]])

    # Load OCC metadata for pc_range/voxel_size
    meta_path = os.path.join(run_dir, "OCC", "occ_metadata.json")
    with open(meta_path) as f:
        occ_meta = json.load(f)
    pc_range = occ_meta.get("pc_range", [-50, -50, -5, 50, 50, 3])
    voxel_size = occ_meta.get("voxel_size", [0.5, 0.5, 0.5])

    from tqdm import tqdm

    sample_step = sample
    dot_radius = 2  # pixels

    for occ_fname in tqdm(occ_files, desc="Projecting OCC"):
        frame = int(occ_fname.replace(".npy", ""))
        if frame_number is not None and frame != frame_number:
            continue

        # Find nearest camera image
        cam_files = sorted(os.listdir(cam_img_dir)) if os.path.isdir(cam_img_dir) else []
        cam_frames = [int(f.replace(".jpg", "").replace(".npy", "")) for f in cam_files if f.endswith((".jpg",".npy"))]
        nearest_cam = min(cam_frames, key=lambda x: abs(x - frame)) if cam_frames else None
        if nearest_cam is None or abs(nearest_cam - frame) > 2:
            print(f"Frame {frame}: no nearby camera image found")
            continue

        # Load camera image
        cam_path = None
        for ext in [".jpg", ".npy"]:
            p = os.path.join(cam_img_dir, f"{nearest_cam:08d}{ext}")
            if os.path.exists(p):
                cam_path = p
                break
        if cam_path is None:
            continue
        if cam_path.endswith(".npy"):
            cam_img = np.load(cam_path)[:, :, :3]
        else:
            cam_img = np.array(Image.open(cam_path))
        if cam_img.shape[2] == 4:
            cam_img = cam_img[:, :, :3]

        # Get ego pose and camera world transform
        if frame not in poses:
            print(f"Frame {frame}: no ego pose")
            continue
        pose_data = poses[frame]
        ex, ey, ez, er, ep, eyaw = pose_data[:6]
        if len(pose_data) >= 12:
            cx, cy, cz, cr, cp, cyaw = pose_data[6:12]
        else:
            # Camera world = ego world + mount (approximate)
            er_rad, ep_rad, eyaw_rad = m.radians(er), m.radians(ep), m.radians(eyaw)
            cyaw_cos, cyaw_sin = m.cos(eyaw_rad), m.sin(eyaw_rad)
            cx = ex + mx * cyaw_cos - my * cyaw_sin
            cy = ey + mx * cyaw_sin + my * cyaw_cos
            cz = ez + mz
            cr, cp = er + mroll, ep + mpitch
            cyaw = eyaw + myaw_m

        # Load OCC
        occ = np.load(os.path.join(occ_npy_dir, occ_fname))

        # Project OCC → camera image
        # Build camera world→image transform
        cr_rad, cp_rad, cyaw_rad = m.radians(cr), m.radians(cp), m.radians(cyaw)

        # World → camera rotation (inverse of camera's world rotation)
        # CARLA composes: R_world = R_yaw(yaw) @ R_pitch(pitch) @ R_roll(roll)
        # Inverse: R_cam = R_roll(-roll) @ R_pitch(-pitch) @ R_yaw(-yaw)
        cr2, sr2 = m.cos(-cyaw_rad), m.sin(-cyaw_rad)
        cp2, sp2 = m.cos(-cp_rad), m.sin(-cp_rad)
        crr, srr = m.cos(-cr_rad), m.sin(-cr_rad)
        R_yaw_inv = np.array([[cr2, -sr2, 0], [sr2, cr2, 0], [0, 0, 1]])
        R_pitch_inv = np.array([[cp2, 0, sp2], [0, 1, 0], [-sp2, 0, cp2]])
        R_roll_inv = np.array([[1, 0, 0], [0, crr, -srr], [0, srr, crr]])
        R_cam = R_yaw_inv @ R_pitch_inv @ R_roll_inv  # world→camera: yaw⁻¹ @ pitch⁻¹ @ roll⁻¹

        cam_pos = np.array([cx, cy, cz])

        # Get all occupied voxels (use --sample to control density)
        xs, ys, zs_n = np.where(occ > 0)
        mask = (xs % sample_step == 0) & (ys % sample_step == 0) & (zs_n % sample_step == 0)
        xs, ys, zs_n = xs[mask], ys[mask], zs_n[mask]
        tags = occ[xs, ys, zs_n]
        if len(xs) == 0:
            continue

        # Ego frame voxel centers
        vx_e = pc_range[0] + (xs + 0.5) * voxel_size[0]  # forward
        vy_e = pc_range[1] + (ys + 0.5) * voxel_size[1]  # left
        vz_e = pc_range[2] + (zs_n + 0.5) * voxel_size[2]  # up

        # Ego frame → CARLA world
        eyaw_rad2 = m.radians(eyaw)
        cy_e, sy_e = m.cos(eyaw_rad2), m.sin(eyaw_rad2)
        # Ego (fwd=X, left=Y) → world (fwd=X, right=Y): left→right=-left
        wx = ex + vx_e * cy_e + vy_e * sy_e
        wy = ey + vx_e * sy_e - vy_e * cy_e
        wz = ez + vz_e

        # World → camera
        pts_world = np.stack([wx, wy, wz], axis=1)  # (N, 3)
        pts_cam = (R_cam @ (pts_world - cam_pos).T).T  # (N, 3)
        # Camera: X=forward, Y=right, Z=up
        Xc, Yc, Zc = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

        # In front of camera
        valid = Xc > 0.5
        if valid.sum() == 0:
            continue

        Xc, Yc, Zc = Xc[valid], Yc[valid], Zc[valid]
        t = tags[valid]
        dist = Xc  # for depth sorting

        # Project to image: u = fx*(Y/X) + cx, v = fy*(-Z/X) + cy
        u = K["fx"] * (Yc / Xc) + K["cx"]
        v = K["fy"] * (-Zc / Xc) + K["cy"]

        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if in_img.sum() == 0:
            continue
        u, v, t, dist = u[in_img], v[in_img], t[in_img], dist[in_img]
        ui, vi = u.astype(int), v.astype(int)

        # Z-buffer: only keep closest voxel per pixel
        depth_buf = np.full((h, w), np.inf, dtype=np.float32)
        result = cam_img.copy()

        # Sort near→far so closer voxels are drawn first (and occlude farther ones)
        order = np.argsort(dist)
        ui, vi, t, dist = ui[order], vi[order], t[order], dist[order]

        r = dot_radius
        for i in range(len(ui)):
            tag = t[i]
            color = LABEL_COLORS.get(int(tag), (128, 128, 128))
            d = dist[i]
            r0, r1 = max(0, vi[i]-r), min(h-1, vi[i]+r)
            c0, c1 = max(0, ui[i]-r), min(w-1, ui[i]+r)
            # Only draw where this voxel is closer than what's already drawn
            patch_depth = depth_buf[r0:r1+1, c0:c1+1]
            mask = d < patch_depth
            if mask.any():
                yy, xx = np.where(mask)
                result[r0+yy, c0+xx] = color
                depth_buf[r0+yy, c0+xx] = d

        out_path = os.path.join(out_dir, f"{frame:08d}.jpg")
        Image.fromarray(result).save(out_path)
        tqdm.write(f"Frame {frame}: {valid.sum()} pts → {os.path.basename(out_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Project OCC voxels onto camera image for validation")
    parser.add_argument("run_dir", help="Path to collection run directory")
    parser.add_argument("--frame", type=int, default=None,
                        help="Specific frame to process (default: all)")
    parser.add_argument("--sample", type=int, default=1,
                        help="Voxel downsampling step (default: 1)")
    parser.add_argument("--channel", default="CAM_FRONT",
                        help="Camera channel to project onto (default: CAM_FRONT)")
    args = parser.parse_args()
    run(args.run_dir, args.frame, args.sample, args.channel)


if __name__ == "__main__":
    main()
