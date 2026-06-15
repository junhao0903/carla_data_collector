#!/usr/bin/env python3
"""Project OCC voxels onto camera image or virtual overhead view.

Usage:
    python tools/occ_projection.py output/<run>                  # virtual overhead
    python tools/occ_projection.py output/<run> --channel CAM_FRONT  # camera
"""
import argparse
import csv
import json
import math as m
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

LABEL_COLORS = {
    0: (0, 0, 0), 1: (128, 64, 128), 2: (244, 35, 232),
    3: (70, 70, 70), 4: (102, 102, 156), 5: (190, 153, 153),
    6: (153, 153, 153), 8: (220, 220, 0), 9: (107, 142, 35),
    12: (220, 20, 60), 14: (0, 0, 142), 15: (0, 0, 70),
    16: (0, 60, 100), 18: (0, 0, 230), 19: (119, 11, 32),
}


def load_ego_poses(ego_csv):
    poses = {}
    with open(ego_csv) as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            d = (float(row["x"]), -float(row["y_left"]), float(row["z"]),
                 float(row["roll_left"]), -float(row["pitch"]), -float(row["yaw_left"]))
            if "cam_x" in row:
                d += (float(row["cam_roll_left"]), -float(row["cam_pitch"]),
                      -float(row["cam_yaw_left"]))
            poses[frame] = d
    return poses


def _build_lookat_R(look_dir):
    """Build camera→world rotation matrix from look-at direction in ego frame."""
    fwd = np.array(look_dir, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    return np.column_stack([fwd, right, up2])  # camera→world


def _project_points(pts_ego, tags, K, cam_pos, R_w2c):
    """Project ego-frame points to image pixels.

    Args:
        pts_ego: (N, 3) in ego/world frame
        tags: (N,) uint8
        K: intrinsics dict
        cam_pos: (3,) camera position in same frame as pts_ego
        R_w2c: 3×3 world→camera rotation

    Returns:
        (ui, vi, tags, dist) or None
    """
    pts_cam = (R_w2c @ (pts_ego - cam_pos).T).T  # (N, 3) camera frame
    Xc, Yc, Zc = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid = Xc > 0.5
    if valid.sum() == 0:
        return None
    Xc, Yc, Zc = Xc[valid], Yc[valid], Zc[valid]
    t = tags[valid]
    u = K["fx"] * (Yc / Xc) + K["cx"]
    v = K["fy"] * (-Zc / Xc) + K["cy"]
    in_img = (u >= 0) & (u < K["width"]) & (v >= 0) & (v < K["height"])
    if in_img.sum() == 0:
        return None
    u, v, t, dist = u[in_img], v[in_img], t[in_img], Xc[in_img]
    return u.astype(int), v.astype(int), t, dist


def _draw_z_buffer(ui, vi, tags, dist, K, bg_img=None):
    w, h = K["width"], K["height"]
    depth_buf = np.full((h, w), np.inf, dtype=np.float32)
    result = bg_img.copy() if bg_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
    order = np.argsort(dist)
    ui, vi, tags, dist = ui[order], vi[order], tags[order], dist[order]
    r = 2
    for i in range(len(ui)):
        color = LABEL_COLORS.get(int(tags[i]), (128, 128, 128))
        d = dist[i]
        r0, r1 = max(0, vi[i]-r), min(h-1, vi[i]+r)
        c0, c1 = max(0, ui[i]-r), min(w-1, ui[i]+r)
        patch = depth_buf[r0:r1+1, c0:c1+1]
        mask = d < patch
        if mask.any():
            yy, xx = np.where(mask)
            result[r0+yy, c0+xx] = color
            depth_buf[r0+yy, c0+xx] = d
    return result


def _load_occ_meta(run_dir):
    meta_path = os.path.join(run_dir, "OCC", "occ_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return (meta.get("pc_range", [-50, -50, -5, 50, 50, 10]),
            meta.get("voxel_size", [0.5, 0.5, 0.5]))


def _sample_occ_points(occ, pc_range, voxel_size, sample):
    xs, ys, zs_n = np.where(occ > 0)
    mask = (xs % sample == 0) & (ys % sample == 0) & (zs_n % sample == 0)
    xs, ys, zs_n = xs[mask], ys[mask], zs_n[mask]
    if len(xs) == 0:
        return None
    pts = np.stack([
        pc_range[0] + (xs + 0.5) * voxel_size[0],
        pc_range[1] + (ys + 0.5) * voxel_size[1],
        pc_range[2] + (zs_n + 0.5) * voxel_size[2],
    ], axis=1).astype(np.float32)
    return pts, occ[xs, ys, zs_n]


def load_camera_intrinsics(run_dir, channel):
    layout_path = os.path.join(run_dir, "sensor_layout.yaml")
    if os.path.exists(layout_path):
        import yaml
        with open(layout_path) as f:
            for s in yaml.safe_load(f).get("sensors", []):
                if s.get("channel") == channel and s.get("modality") == "camera_rgb":
                    out = s.get("output", {})
                    w, h = out.get("width", 1600), out.get("height", 900)
                    if "fx" in out:
                        return {"fx": out["fx"], "fy": out["fy"], "cx": out["cx"], "cy": out["cy"],
                                "width": w, "height": h}
                    hfov = m.radians(out.get("fov", 70))
                    fx = w / (2 * m.tan(hfov / 2))
                    vfov = 2 * m.atan(m.tan(hfov / 2) * h / w)
                    fy = h / (2 * m.tan(vfov / 2))
                    return {"fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2,
                            "width": w, "height": h}
    return {"fx": 1600, "fy": 1600, "cx": 800, "cy": 450, "width": 1600, "height": 900}


# ── Virtual overhead view ──

VIRTUAL_POS = (0.0, 0.0, 30.0)       # ego frame (fwd, left, up)
VIRTUAL_PITCH = -45.0                  # degrees, look-down angle
VIRTUAL_YAW = 0.0
VIRTUAL_FOV = 90
VIRTUAL_W, VIRTUAL_H = 800, 800


def _run_virtual_view(run_dir, occ_files, frame_number, sample):
    hfov = m.radians(VIRTUAL_FOV)
    fx = VIRTUAL_W / (2 * m.tan(hfov / 2))
    K = {"fx": fx, "fy": fx, "cx": VIRTUAL_W / 2, "cy": VIRTUAL_H / 2,
         "width": VIRTUAL_W, "height": VIRTUAL_H}
    cam_pos = np.array(VIRTUAL_POS, dtype=np.float32)
    # Look direction: forward + down
    look_dir = (m.cos(m.radians(VIRTUAL_PITCH)) * m.cos(m.radians(VIRTUAL_YAW)),
                m.cos(m.radians(VIRTUAL_PITCH)) * m.sin(m.radians(VIRTUAL_YAW)),
                m.sin(m.radians(VIRTUAL_PITCH)))
    R_c2w = _build_lookat_R(look_dir)
    R_w2c = R_c2w.T  # ego→camera

    out_dir = os.path.join(run_dir, "OCC", "projection")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Virtual overhead: pos={VIRTUAL_POS}, pitch={VIRTUAL_PITCH}°, "
          f"yaw={VIRTUAL_YAW}°, fov={VIRTUAL_FOV}°")

    pc_range, voxel_size = _load_occ_meta(run_dir)
    for occ_fname in tqdm(occ_files, desc="Projecting overhead"):
        frame = int(occ_fname.replace(".npy", ""))
        if frame_number is not None and frame != frame_number:
            continue
        occ = np.load(os.path.join(run_dir, "OCC", "original", occ_fname))
        sampled = _sample_occ_points(occ, pc_range, voxel_size, sample)
        if sampled is None:
            continue
        pts_ego, tags = sampled
        proj = _project_points(pts_ego, tags, K, cam_pos, R_w2c)
        if proj is None:
            continue
        result = _draw_z_buffer(*proj, K)
        Image.fromarray(result).save(os.path.join(out_dir, f"{frame:08d}.jpg"))


# ── Camera projection ──

def _run_camera_projection(run_dir, occ_files, channel, frame_number, sample):
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if not os.path.exists(ego_csv):
        print("No ego_trajectory.csv found")
        return
    poses = load_ego_poses(ego_csv)
    K = load_camera_intrinsics(run_dir, channel)
    cam_img_dir = os.path.join(run_dir, channel, "original")
    out_dir = os.path.join(run_dir, "OCC", f"projection_{channel}")
    os.makedirs(out_dir, exist_ok=True)
    pc_range, voxel_size = _load_occ_meta(run_dir)

    for occ_fname in tqdm(occ_files, desc=f"Projecting {channel}"):
        frame = int(occ_fname.replace(".npy", ""))
        if frame_number is not None and frame != frame_number:
            continue

        cam_files = sorted(os.listdir(cam_img_dir)) if os.path.isdir(cam_img_dir) else []
        cam_frames = [int(f.replace(".jpg", "").replace(".npy", ""))
                      for f in cam_files if f.endswith((".jpg", ".npy"))]
        nearest = min(cam_frames, key=lambda x: abs(x - frame)) if cam_frames else None
        if nearest is None or abs(nearest - frame) > 2:
            continue
        cam_path = None
        for ext in [".jpg", ".npy"]:
            p = os.path.join(cam_img_dir, f"{nearest:08d}{ext}")
            if os.path.exists(p):
                cam_path = p; break
        if cam_path is None:
            continue
        cam_img = np.array(Image.open(cam_path))[:, :, :3]

        if frame not in poses:
            continue
        ex, ey, ez, er, ep, eyaw = poses[frame][:6]
        if len(poses[frame]) >= 9:
            cr, cp, cyaw = poses[frame][6:9]
        else:
            cr, cp, cyaw = er, ep, eyaw

        occ = np.load(os.path.join(run_dir, "OCC", "original", occ_fname))
        sampled = _sample_occ_points(occ, pc_range, voxel_size, sample)
        if sampled is None:
            continue
        pts_ego, tags = sampled

        # Ego frame → world
        vx, vy, vz = pts_ego[:, 0], pts_ego[:, 1], pts_ego[:, 2]
        eyaw_rad = m.radians(eyaw)
        cy_e, sy_e = m.cos(eyaw_rad), m.sin(eyaw_rad)
        wx = ex + vx * cy_e + vy * sy_e
        wy = ey + vx * sy_e - vy * cy_e
        wz = ez + vz
        pts_world = np.stack([wx, wy, wz], axis=1)

        # Camera world position
        if len(poses[frame]) >= 9:
            cam_pos = np.array([poses[frame][6], poses[frame][7], poses[frame][8]])
        else:
            cam_pos = np.array([ex, ey, ez])
        cr_rad, cp_rad, cy_rad = m.radians(cr), m.radians(cp), m.radians(cyaw)

        # World → camera rotation (inverse)
        cr2, sr2 = m.cos(-cy_rad), m.sin(-cy_rad)
        cp2, sp2 = m.cos(-cp_rad), m.sin(-cp_rad)
        crr, srr = m.cos(-cr_rad), m.sin(-cr_rad)
        R_yaw_inv = np.array([[cr2, -sr2, 0], [sr2, cr2, 0], [0, 0, 1]])
        R_pit_inv = np.array([[cp2, 0, sp2], [0, 1, 0], [-sp2, 0, cp2]])
        R_rol_inv = np.array([[1, 0, 0], [0, crr, -srr], [0, srr, crr]])
        R_w2c = R_yaw_inv @ R_pit_inv @ R_rol_inv

        proj = _project_points(pts_world, tags, K, cam_pos, R_w2c)
        if proj is None:
            continue
        result = _draw_z_buffer(*proj, K, cam_img)
        Image.fromarray(result).save(os.path.join(out_dir, f"{frame:08d}.jpg"))


# ── Entry ──

def run(run_dir, frame_number=None, sample=1, channel=None):
    occ_dir = os.path.join(run_dir, "OCC", "original")
    if not os.path.isdir(occ_dir):
        print(f"No OCC npy files in {occ_dir}")
        return
    occ_files = sorted(f for f in os.listdir(occ_dir) if f.endswith(".npy"))
    if not occ_files:
        print("No .npy files found")
        return
    if channel:
        _run_camera_projection(run_dir, occ_files, channel, frame_number, sample)
    else:
        _run_virtual_view(run_dir, occ_files, frame_number, sample)


def main():
    parser = argparse.ArgumentParser(
        description="Project OCC voxels onto camera image or virtual overhead view")
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Path to collection run directory (default: latest output)")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--channel", default=None,
                        help="Camera channel (default: virtual overhead view)")
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        import glob
        dirs = sorted(glob.glob("output/*"))
        if not dirs:
            print("No output directories found")
            return
        run_dir = dirs[-1]
        print(f"Using latest: {run_dir}")
    run(run_dir, args.frame, args.sample, args.channel)


if __name__ == "__main__":
    main()
