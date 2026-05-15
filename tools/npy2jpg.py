#!/usr/bin/env python3
"""Post-process run directory: convert npy to jpg, generate visualizations, build OCC."""
import sys
import os
import glob
import json
import math as m
import numpy as np
from PIL import Image

# CARLA semantic colors for OCC viz
OCC_COLORS = {
    0: (0, 0, 0),          # unknown
    1: (128, 64, 128),     # road
    2: (244, 35, 232),     # sidewalk
    3: (70, 70, 70),       # building
    4: (102, 102, 156),    # wall
    5: (190, 153, 153),    # fence
    6: (153, 153, 153),    # pole
    7: (250, 170, 30),     # traffic_light
    8: (220, 220, 0),      # traffic_sign
    9: (107, 142, 35),     # vegetation
    10: (152, 251, 152),   # terrain
    11: (70, 130, 180),    # sky
    12: (220, 20, 60),     # pedestrian
    13: (255, 0, 0),       # rider
    14: (0, 0, 142),       # car
    15: (0, 0, 70),        # truck
    16: (0, 60, 100),      # bus
    17: (0, 80, 100),      # train
    18: (0, 0, 230),       # motorcycle
    19: (119, 11, 32),     # bicycle
    20: (110, 190, 160),   # static
    21: (170, 120, 50),    # dynamic
    22: (55, 90, 80),      # other
    23: (45, 60, 150),     # water
    24: (157, 234, 50),    # road_line
    25: (81, 0, 81),       # ground
    26: (150, 100, 100),   # bridge
    27: (230, 150, 140),   # rail_track
    28: (180, 165, 180),   # guard_rail
}


def convert_run(run_dir, quality=95):
    _convert_orin(run_dir, quality)
    _depth_visualization(run_dir, quality)
    _semantic_visualization(run_dir)
    _annotation_visualization(run_dir)
    _generate_occ(run_dir)


def _convert_orin(run_dir, quality):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        sub_dir = os.path.join(cam_dir, "original")
        if not os.path.isdir(sub_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(sub_dir, "*.npy")))
        if not npy_files:
            continue
        print(f"{cam_dir}/original: {len(npy_files)} files -> jpg")
        for npy_path in npy_files:
            arr = np.load(npy_path)
            if arr.ndim == 3 and arr.shape[2] == 4:
                jpg_path = npy_path.replace(".npy", ".jpg")
                Image.fromarray(arr[:, :, [2, 1, 0]]).save(jpg_path, quality=quality)  # BGRA→RGB
                os.remove(npy_path)


def _depth_visualization(run_dir, quality):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        src_dir = os.path.join(cam_dir, "depth")
        if not os.path.isdir(src_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
        if not npy_files:
            continue
        viz_dir = os.path.join(cam_dir, "depth_viz")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"{cam_dir}/depth_viz: {len(npy_files)} files (npy kept in depth/)")
        for npy_path in npy_files:
            depth = np.load(npy_path)
            clipped = np.clip(depth, 0.0, 250.0)
            normalized = 1.0 - clipped / 250.0
            normalized = normalized ** 0.4
            gray = (normalized * 255).astype(np.uint8)
            fname = os.path.basename(npy_path).replace(".npy", ".png")
            Image.fromarray(gray, mode="L").save(os.path.join(viz_dir, fname))


# ========== OCC Post-Processing ==========

def _load_grid_params(run_dir):
    occ_dir = os.path.join(run_dir, "OCC_GT")
    cfg_path = os.path.join(occ_dir, "grid_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        return (cfg["x_min_m"], cfg["x_max_m"], cfg["y_min_m"], cfg["y_max_m"],
                cfg.get("z_min_m", -2), cfg.get("z_max_m", 4), cfg["resolution_m"])
    return (-20, 80, -40, 40, -2, 4, 0.5)


def _load_ego_poses(run_dir):
    poses = {}
    path = os.path.join(run_dir, "ego_trajectory.csv")
    if not os.path.exists(path):
        return poses
    with open(path) as f:
        header = next(f).strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            try:
                p = {"x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3]),
                     "roll": float(parts[4]), "pitch": float(parts[5]), "yaw": float(parts[6])}
                # Camera transforms (columns 7-12 if present)
                if len(parts) >= 13:
                    p["cam_x"] = float(parts[7])
                    p["cam_y"] = float(parts[8])
                    p["cam_z"] = float(parts[9])
                    p["cam_roll"] = float(parts[10])
                    p["cam_pitch"] = float(parts[11])
                    p["cam_yaw"] = float(parts[12])
                poses[int(parts[0])] = p
            except (ValueError, IndexError):
                pass
    return poses


def _label_lidar_with_camera(run_dir, lidar_points, frame, ego_poses):
    """Project LiDAR points onto semantic camera using real camera transforms."""
    tags = np.full(len(lidar_points), 20, dtype=np.uint8)
    if frame not in ego_poses:
        return tags
    ego = ego_poses[frame]
    cam_wx = ego.get("cam_x")
    if cam_wx is None:
        return tags
    cam_wy, cam_wz = ego["cam_y"], ego["cam_z"]
    cam_yaw = m.radians(ego.get("cam_yaw", ego["yaw"]))
    cam_pitch = m.radians(ego.get("cam_pitch", 0))
    cam_roll = m.radians(ego.get("cam_roll", 0))

    # Find nearest semantic image
    sem_img = None
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        sem_dir = os.path.join(cam_dir, "semantic")
        if not os.path.isdir(sem_dir):
            continue
        sem_files = sorted(os.listdir(sem_dir))
        if not sem_files:
            continue
        sem_frames = [int(f.replace(".png", "")) for f in sem_files if f.endswith(".png")]
        if not sem_frames:
            continue
        nearest = min(sem_frames, key=lambda x: abs(x - frame))
        if abs(nearest - frame) > 5:
            continue
        sem_img = np.array(Image.open(os.path.join(sem_dir, f"{nearest:08d}.png")))
        break
    if sem_img is None:
        return tags
    h, w = sem_img.shape

    # Camera intrinsic (consistent with sensor config)
    fov = m.radians(70)
    fx = w / (2 * m.tan(fov / 2))
    fy = h / (2 * m.tan(fov / 2))
    cx, cy = w / 2, h / 2

    # Camera → world rotation matrix
    cy = m.cos(cam_yaw); sy = m.sin(cam_yaw)
    cp = m.cos(cam_pitch); sp = m.sin(cam_pitch)
    cr = m.cos(cam_roll); sr = m.sin(cam_roll)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_cam = R_yaw @ R_pitch @ R_roll  # camera→world rotation

    # World → camera: R_cam^T @ (p_world - cam_pos)
    px = lidar_points[:, 0] - cam_wx
    py = lidar_points[:, 1] - cam_wy
    pz = lidar_points[:, 2] - cam_wz
    pts_world = np.stack([px, py, pz], axis=-1)  # (N, 3)
    pts_cam = (R_cam.T @ pts_world.T).T  # (N, 3)

    X = pts_cam[:, 0]  # forward
    Y = pts_cam[:, 1]  # right
    Z = pts_cam[:, 2]  # up

    front = X > 0.1
    if front.sum() == 0:
        return tags

    u = (fx * Y[front] / X[front] + cx).astype(np.int32)
    v = (fy * (-Z[front]) / X[front] + cy).astype(np.int32)
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if in_img.sum() == 0:
        return tags
    u, v = u[in_img], v[in_img]
    idx = np.where(front)[0][in_img]
    sampled = sem_img[v, u]
    valid_label = sampled > 0  # skip tag 0 (unlabeled), keep default 20
    tags[idx[valid_label]] = sampled[valid_label]
    return tags


def _semantic_visualization(run_dir):
    """Generate colorized semantic PNGs in semantic_viz/"""
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        src_dir = os.path.join(cam_dir, "semantic")
        if not os.path.isdir(src_dir):
            continue
        png_files = sorted(glob.glob(os.path.join(src_dir, "*.png")))
        if not png_files:
            continue
        viz_dir = os.path.join(cam_dir, "semantic_viz")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"{cam_dir}/semantic_viz: {len(png_files)} files")
        for png_path in png_files:
            tags = np.array(Image.open(png_path))  # (H, W) uint8
            rgb = np.zeros((tags.shape[0], tags.shape[1], 3), dtype=np.uint8)
            for tag, color in OCC_COLORS.items():
                rgb[tags == tag] = color
            fname = os.path.basename(png_path)
            Image.fromarray(rgb).save(os.path.join(viz_dir, fname))


def _annotation_visualization(run_dir):
    """Draw 2D bounding boxes on camera images."""
    import json
    # Category colors (BGR for PIL drawing)
    COLORS = {"vehicle": (0, 255, 0), "pedestrian": (0, 0, 255)}

    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        orin_dir = os.path.join(cam_dir, "original")
        ann_dir = os.path.join(cam_dir, "annotations")
        if not os.path.isdir(orin_dir) or not os.path.isdir(ann_dir):
            continue
        viz_dir = os.path.join(cam_dir, "annotations_viz")
        os.makedirs(viz_dir, exist_ok=True)
        img_files = sorted(glob.glob(os.path.join(orin_dir, "*.jpg")))
        if not img_files:
            continue
        print(f"{cam_dir}/annotations_viz: {len(img_files)} files")
        for img_path in img_files:
            fname = os.path.basename(img_path)
            frame = fname.replace(".jpg", "")
            ann_path = os.path.join(ann_dir, f"{frame}.json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path) as f:
                anns = json.load(f)
            if not anns:
                continue
            from PIL import ImageDraw
            img = Image.open(img_path).copy()
            draw = ImageDraw.Draw(img)
            for a in anns:
                bbox = a.get("bbox_2d")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                color = COLORS.get(a.get("category", "vehicle"), (0, 255, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label = f"{a['category']}_{a['actor_id']}"
                draw.text((x1, y1 - 10), label, fill=color)
            img.save(os.path.join(viz_dir, fname))


def _generate_occ(run_dir):
    """Post-process: generate OCC from LiDAR + actor annotations."""
    x_min, x_max, y_min, y_max, z_min, z_max, res = _load_grid_params(run_dir)
    occ_dir = os.path.join(run_dir, "OCC_GT")
    os.makedirs(occ_dir, exist_ok=True)
    with open(os.path.join(occ_dir, "metadata.json"), "w") as f:
        json.dump({"x_min_m": x_min, "x_max_m": x_max, "y_min_m": y_min, "y_max_m": y_max,
                    "z_min_m": z_min, "z_max_m": z_max, "resolution_m": res}, f)

    ego_poses = _load_ego_poses(run_dir)
    nx = int(round((x_max - x_min) / res))
    ny = int(round((y_max - y_min) / res))
    nz = int(round((z_max - z_min) / res))

    # === LiDAR → OCC ===
    lidar_dir = os.path.join(run_dir, "LIDAR_TOP", "original")
    if os.path.isdir(lidar_dir):
        lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.npy")))
        print(f"OCC from LiDAR: {len(lidar_files)} frames → {occ_dir}")
        for lpath in lidar_files:
            frame_str = os.path.basename(lpath).replace(".npy", "")
            frame = int(frame_str)
            if frame not in ego_poses:
                continue
            occ_path = os.path.join(occ_dir, f"{frame_str}.npy")
            points = np.load(lpath)
            # Transform to ego frame
            ego = ego_poses[frame]
            eyaw = m.radians(ego["yaw"])
            cy, sy = m.cos(eyaw), m.sin(eyaw)
            px, py, pz = points[:,0]-ego["x"], points[:,1]-ego["y"], points[:,2]-ego["z"]
            ex = cy*px + sy*py
            ey = -sy*px + cy*py
            ez = pz
            ix = np.floor((ex-x_min)/res).astype(np.int32)
            iy = np.floor((ey-y_min)/res).astype(np.int32)
            iz = np.floor((ez-z_min)/res).astype(np.int32)
            valid = (ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)&(iz>=0)&(iz<nz)
            if valid.sum() == 0:
                continue
            ix, iy, iz = ix[valid], iy[valid], iz[valid]
            tags = _label_lidar_with_camera(run_dir, points, frame, ego_poses)
            tags = tags[valid]
            # Deduplicate at OCC resolution (keep last tag per voxel)
            occ_idx = ix * ny * nz + iy * nz + iz
            _, ui = np.unique(occ_idx, return_index=True)
            ix, iy, iz, tags = ix[ui], iy[ui], iz[ui], tags[ui]
            grid = np.zeros((nz, ny, nx), dtype=np.uint8)
            grid[iz, iy, ix] = tags
            np.save(occ_path, grid)

    # === Actor annotations → overlaid on OCC ===
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        ann_dir = os.path.join(cam_dir, "annotations")
        if not os.path.isdir(ann_dir):
            continue
        ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
        if not ann_files:
            continue
        print(f"Overlay actors: {len(ann_files)} frames")
        actor_map = {"vehicle": 14, "pedestrian": 12}
        for ann_path in ann_files:
            frame = int(os.path.basename(ann_path).replace(".json", ""))
            if frame not in ego_poses:
                continue
            occ_path = os.path.join(occ_dir, f"{frame:08d}.npy")
            if not os.path.exists(occ_path):
                continue
            with open(ann_path) as f:
                anns = json.load(f)
            if not anns:
                continue
            grid = np.load(occ_path)
            ego = ego_poses[frame]
            eyaw = m.radians(ego["yaw"])
            cy, sy = m.cos(eyaw), m.sin(eyaw)
            for a in anns:
                cat = actor_map.get(a.get("category"), 21)
                loc, rot, bb = a["location"], a["rotation"], a["bbox_3d"]
                dx, dy, dz = loc["x"]-ego["x"], loc["y"]-ego["y"], loc["z"]-ego["z"]
                rel_y = m.radians(rot["yaw"]) - eyaw
                cr, sr = m.cos(rel_y), m.sin(rel_y)
                hx, hy, hz = bb["x"]/2, bb["y"]/2, bb["z"]/2
                cx = cy*dx + sy*dy
                cz = cy*dy - sy*dx  # wait this is wrong - let me fix
                # Actually: cy = ego y in ego frame
                cx = cy*dx + sy*dy
                cy_e = -sy*dx + cy*dy
                cz_e = dz
                corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
                rot_m = np.array([[cr, -sr], [sr, cr]])
                corners = np.dot(corners, rot_m.T)
                corners[:,0] += cx
                corners[:,1] += cy_e
                zl, zh = cz_e - hz, cz_e + hz
                x1 = max(0, int(m.floor((np.min(corners[:,0])-x_min)/res)))
                x2 = min(nx-1, int(m.floor((np.max(corners[:,0])-x_min)/res)))
                y1 = max(0, int(m.floor((np.min(corners[:,1])-y_min)/res)))
                y2 = min(ny-1, int(m.floor((np.max(corners[:,1])-y_min)/res)))
                z1 = max(0, int(m.floor((zl-z_min)/res)))
                z2 = min(nz-1, int(m.floor((zh-z_min)/res)))
                if x2<x1 or y2<y1 or z2<z1:
                    continue
                for iz in range(z1, z2+1):
                    for iy in range(y1, y2+1):
                        for ix in range(x1, x2+1):
                            px = x_min+(ix+.5)*res; py = y_min+(iy+.5)*res
                            ok = True
                            for ei in range(4):
                                a = corners[ei]; b = corners[(ei+1)%4]
                                if (b[0]-a[0])*(py-a[1])-(b[1]-a[1])*(px-a[0]) > 0:
                                    ok = False; break
                            if ok:
                                grid[iz, iy, ix] = cat
            np.save(occ_path, grid)

    # === Generate BEV visualization ===
    _occ_visualization(run_dir, occ_dir, x_min, x_max, y_min, y_max, res)


def _occ_visualization(run_dir, occ_dir, x_min, x_max, y_min, y_max, res):
    npy_files = sorted(glob.glob(os.path.join(occ_dir, "*.npy")))
    if not npy_files:
        return
    viz_dir = os.path.join(run_dir, "OCC_GT_viz")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"OCC_GT_viz: {len(npy_files)} files")
    for npy_path in npy_files:
        grid = np.load(npy_path)
        bev = grid.max(axis=0)
        bev_img = np.flipud(bev.T)
        rgb = np.zeros((bev_img.shape[0], bev_img.shape[1], 3), dtype=np.uint8)
        for cat, color in OCC_COLORS.items():
            rgb[bev_img == cat] = color
        scale = 6
        # Crop square centered on ego, using min available range
        half = min(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
        r1 = max(0, int((x_max - half) / res))
        r2 = min(bev_img.shape[0], int((x_max + half) / res))
        c1 = max(0, int((-half - y_min) / res))
        c2 = min(bev_img.shape[1], int((half - y_min) / res))
        bev_crop = bev_img[r1:r2, c1:c2]
        rgb = np.zeros((bev_crop.shape[0], bev_crop.shape[1], 3), dtype=np.uint8)
        for cat, color in OCC_COLORS.items():
            rgb[bev_crop == cat] = color
        rgb_big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        # Ego at crop center
        ego_r = rgb_big.shape[0] // 2
        ego_c = rgb_big.shape[1] // 2
        h, w = rgb_big.shape[:2]
        if 0 <= ego_r < h and 0 <= ego_c < w:
            rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            rgb_big[(rr-ego_r)**2 + (cc-ego_c)**2 < (4*scale)**2] = (0, 255, 0)
        fname = os.path.basename(npy_path).replace(".npy", ".png")
        Image.fromarray(rgb_big).save(os.path.join(viz_dir, fname))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/npy2jpg.py output/<run_dir>")
        sys.exit(1)
    convert_run(sys.argv[1])
