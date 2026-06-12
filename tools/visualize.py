#!/usr/bin/env python3
"""Post-run visualization: BEV, depth, semantic, annotations, trajectory."""
import sys, os, glob, json, math as m
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# CARLA semantic colors
OCC_COLORS = {
    0: (0, 0, 0), 1: (128, 64, 128), 2: (244, 35, 232), 3: (70, 70, 70),
    4: (102, 102, 156), 5: (190, 153, 153), 6: (153, 153, 153),
    7: (250, 170, 30), 8: (220, 220, 0), 9: (107, 142, 35),
    10: (152, 251, 152), 11: (70, 130, 180), 12: (220, 20, 60),
    13: (255, 0, 0), 14: (0, 0, 142), 15: (0, 0, 70), 16: (0, 60, 100),
    17: (0, 80, 100), 18: (0, 0, 230), 19: (119, 11, 32),
    20: (110, 190, 160), 21: (170, 120, 50), 22: (55, 90, 80),
    23: (45, 60, 150), 24: (157, 234, 50), 25: (81, 0, 81),
    26: (150, 100, 100), 27: (230, 150, 140), 28: (180, 165, 180),
}


def _load_sensor_layout(run_dir):
    path = os.path.join(run_dir, "sensor_layout.yaml")
    if os.path.exists(path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def _cam_vis_enabled(layout, channel, key, default=True):
    for s in layout.get("sensors", []):
        if s.get("channel") == channel:
            return s.get(key, default)
    return default


def convert_run(run_dir, quality=95, force_all=False):
    layout = _load_sensor_layout(run_dir)
    _depth_visualization(run_dir, layout, quality, force_all)
    _semantic_visualization(run_dir, layout, force_all)
    _filter_bev_viz(run_dir)
    _occ_bev_visualization(run_dir)
    _occ_projection_viz(run_dir, layout, force_all)
    _annotation_visualization(run_dir, layout, force_all)
    _trajectory_visualization(run_dir, layout, force_all)


# ══════════════════════════════════════════════════════════════════════
# Depth visualization
# ══════════════════════════════════════════════════════════════════════

def _depth_visualization(run_dir, layout, quality, force_all=False):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        channel = os.path.basename(cam_dir)
        if not force_all and not _cam_vis_enabled(layout, channel, "depth_vis", True):
            continue
        src_dir = os.path.join(cam_dir, "depth")
        if not os.path.isdir(src_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
        if not npy_files:
            continue
        viz_dir = os.path.join(cam_dir, "depth_viz")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"{cam_dir}/depth_viz: {len(npy_files)} files")
        for npy_path in tqdm(npy_files, desc=f"{channel} depth_viz", leave=False):
            depth = np.load(npy_path)
            clipped = np.clip(depth, 0.0, 250.0)
            normalized = 1.0 - clipped / 250.0
            normalized = normalized ** 0.4
            gray = (normalized * 255).astype(np.uint8)
            fname = os.path.basename(npy_path).replace(".npy", ".png")
            Image.fromarray(gray, mode="L").save(os.path.join(viz_dir, fname))


# ══════════════════════════════════════════════════════════════════════
# Semantic visualization
# ══════════════════════════════════════════════════════════════════════

def _semantic_visualization(run_dir, layout, force_all=False):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        channel = os.path.basename(cam_dir)
        if not force_all and not _cam_vis_enabled(layout, channel, "semantic_vis", True):
            continue
        src_dir = os.path.join(cam_dir, "semantic")
        if not os.path.isdir(src_dir):
            continue
        png_files = sorted(glob.glob(os.path.join(src_dir, "*.png")))
        if not png_files:
            continue
        viz_dir = os.path.join(cam_dir, "semantic_viz")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"{cam_dir}/semantic_viz: {len(png_files)} files")
        for png_path in tqdm(png_files, desc=f"{channel} semantic_viz", leave=False):
            tags = np.array(Image.open(png_path))
            rgb = np.zeros((tags.shape[0], tags.shape[1], 3), dtype=np.uint8)
            for tag, color in OCC_COLORS.items():
                rgb[tags == tag] = color
            Image.fromarray(rgb).save(os.path.join(viz_dir, os.path.basename(png_path)))


# ══════════════════════════════════════════════════════════════════════
# Annotation visualization (2D camera + 3D LiDAR BEV)
# ══════════════════════════════════════════════════════════════════════

def _annotation_visualization(run_dir, layout, force_all=False):
    _camera_annotation_viz(run_dir, layout, force_all)
    _lidar_annotation_viz(run_dir, layout, force_all)


def _camera_annotation_viz(run_dir, layout, force_all=False):
    COLORS = {"vehicle": (0, 255, 0), "pedestrian": (0, 0, 255),
              "static_car": (255, 200, 0), "static_truck": (255, 150, 0),
              "static_bus": (255, 100, 0), "static_train": (255, 50, 0),
              "static_motorcycle": (255, 200, 50), "static_bicycle": (255, 200, 100),
              "static_pedestrian": (200, 100, 255)}
    ann_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if not os.path.isdir(ann_dir):
        return

    from tools.occ_projection import load_camera_intrinsics
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    poses = {}
    if os.path.exists(ego_csv):
        with open(ego_csv) as f:
            import csv as _csv
            for row in _csv.DictReader(f):
                poses[int(row["frame"])] = (
                    float(row["x"]), -float(row["y_left"]), float(row["z"]),
                    -float(row["roll_left"]), float(row["pitch"]), -float(row["yaw_left"]))

    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        channel = os.path.basename(cam_dir)
        if not force_all and not _cam_vis_enabled(layout, channel, "annotation_vis", True):
            continue
        orin_dir = os.path.join(cam_dir, "original")
        if not os.path.isdir(orin_dir):
            continue
        viz_dir = os.path.join(cam_dir, "annotations_viz")
        os.makedirs(viz_dir, exist_ok=True)
        img_files = sorted(glob.glob(os.path.join(orin_dir, "*.jpg")) +
                          glob.glob(os.path.join(orin_dir, "*.npy")))
        if not img_files:
            continue
        K = load_camera_intrinsics(run_dir, channel)
        print(f"{cam_dir}/annotations_viz: {len(img_files)} files")
        for img_path in tqdm(img_files, desc=f"{channel} annotations_viz", leave=False):
            fname = os.path.basename(img_path)
            frame = int(fname.replace(".jpg", "").replace(".npy", ""))
            ann_path = os.path.join(ann_dir, f"{frame:08d}.json")
            if not os.path.exists(ann_path):
                continue
            if frame not in poses:
                continue
            ex, ey, ez, er, ep, eyaw = poses[frame]
            with open(ann_path) as f:
                anns = json.load(f)
            if img_path.endswith(".npy"):
                arr = np.load(img_path)
                img = Image.fromarray(arr[:, :, [2, 1, 0]])
            else:
                img = Image.open(img_path).copy()
            draw = ImageDraw.Draw(img)
            for a in anns:
                loc = a["location"]; rot = a["rotation"]; sz = a["bbox_3d"]
                # AD coords → world to camera projection
                # Simplified: project bbox center
                ax, ay, az = loc["x"], -loc["y"], loc["z"]
                dx, dy, dz = ax - ex, ay - ey, az - ez
                eyaw_r = m.radians(eyaw)
                cr, sr = m.cos(eyaw_r), m.sin(eyaw_r)
                cx_ego = dx * cr + dy * sr
                cy_ego = dx * sr - dy * cr
                cz_ego = dz
                cam_X = cx_ego - 1.5
                cam_Y = cy_ego
                cam_Z = cz_ego - 1.6
                if cam_X <= 0.1:
                    continue
                u = int(K["fx"] * cam_Y / cam_X + K["cx"])
                v = int(K["fy"] * (-cam_Z) / cam_X + K["cy"])
                color = COLORS.get(a.get("category", "vehicle"), (0, 255, 0))
                r = 4
                draw.ellipse([u-r, v-r, u+r, v+r], fill=color)
            img.save(os.path.join(viz_dir, fname))


def _lidar_annotation_viz(run_dir, layout, force_all=False):
    if not force_all:
        for s in layout.get("sensors", []):
            if s.get("modality") in ("lidar", "lidar_semantic") and s.get("enabled", True):
                if not s.get("annotation_vis", True):
                    return
                break
    COLORS = {"vehicle": (0, 255, 0), "pedestrian": (0, 0, 255),
              "static_car": (255, 200, 0), "static_truck": (255, 150, 0),
              "static_bus": (255, 100, 0), "static_motorcycle": (255, 200, 50),
              "static_bicycle": (255, 200, 100)}
    ann_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if not os.path.isdir(ann_dir):
        return

    # LiDAR offset lookup from sensor layout (AD coords: X=fwd, Y=left)
    lidar_offsets = {}
    for s in (layout or {}).get("sensors", []):
        if s.get("modality") in ("lidar", "lidar_semantic"):
            t = s.get("transform", {})
            lidar_offsets[s["channel"]] = (
                t.get("x", 0.0), t.get("y", 0.0), t.get("z", 1.8),
                m.radians(t.get("yaw", 0.0)))

    # Load ego poses in AD coords for sensor-local conversion
    ego_ad = {}
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if os.path.exists(ego_csv):
        with open(ego_csv) as f:
            import csv as _csv
            for row in _csv.DictReader(f):
                ego_ad[int(row["frame"])] = (
                    float(row["x"]), float(row["y_left"]), float(row["z"]),
                    float(row["roll_left"]), float(row["pitch"]), float(row["yaw_left"]))

    for lidar_channel in sorted(glob.glob(os.path.join(run_dir, "LIDAR_*"))):
        channel = os.path.basename(lidar_channel)
        lidar_dir = os.path.join(run_dir, channel, "original")
        if not os.path.isdir(lidar_dir):
            continue
        ann_viz_dir = os.path.join(run_dir, channel, "annotations_viz")
        os.makedirs(ann_viz_dir, exist_ok=True)
        lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.npy")))
        rng, scale, vmin, vmax = 80, 2, -3.0, 2.0
        size = int(2 * rng / 0.2)
        print(f"{channel}/annotations_viz: {len(lidar_files)} frames")
        for lpath in tqdm(lidar_files, desc=f"{channel} lidar_viz", leave=False):
            frame_str = os.path.basename(lpath).replace(".npy", "")
            frame = int(frame_str)
            ann_path = os.path.join(ann_dir, f"{frame_str}.json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path) as f:
                anns = json.load(f)
            points = np.load(lpath)
            lx, ly, lz = points[:, 0], points[:, 1], points[:, 2]
            z_valid = (lz > vmin) & (lz < vmax)
            xy_valid = (np.abs(lx) < rng) & (np.abs(ly) < rng)
            valid = z_valid & xy_valid
            lx, ly, lz = lx[valid], ly[valid], lz[valid]
            if len(lx) == 0:
                continue

            # Get ego pose for sensor-local annotation conversion
            ego = ego_ad.get(frame)

            img = np.zeros((size, size), dtype=np.uint8)
            px = ((rng - lx) / (2 * rng) * size).astype(int)
            py = ((rng - ly) / (2 * rng) * size).astype(int)
            px = np.clip(px, 0, size - 1); py = np.clip(py, 0, size - 1)
            lz_clip = np.clip(lz, vmin, vmax)
            intensity = ((lz_clip - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            np.maximum.at(img, (px, py), intensity)
            img_color = np.stack([img, img, img], axis=-1)
            img_big = np.repeat(np.repeat(img_color, scale, axis=0), scale, axis=1)
            h_img, w_img = img_big.shape[:2]
            cy, cx = h_img // 2, w_img // 2
            rr, cc = np.meshgrid(np.arange(h_img), np.arange(w_img), indexing='ij')
            img_big[(rr - cy) ** 2 + (cc - cx) ** 2 < (1 * scale) ** 2] = (0, 255, 0)
            pil_img = Image.fromarray(img_big)
            draw = ImageDraw.Draw(pil_img)
            pix_per_m = scale * size / (2 * rng)
            for a in anns:
                loc = a["location"]; bb = a["bbox_3d"]
                if bb["x"] == 0 or bb["y"] == 0:
                    bb = {"x": 4.0, "y": 2.0, "z": 1.5}
                if ego is not None:
                    ax_ad, ay_ad, az_ad = loc["x"], loc["y"], loc["z"]
                    ex_ad, ey_ad, ez_ad, eroll, epitch, eyaw = ego
                    eyaw_r = m.radians(eyaw)
                    ce, se = m.cos(eyaw_r), m.sin(eyaw_r)
                    # annotation → ego frame (AD: X=fwd, Y=left)
                    dx = ax_ad - ex_ad
                    dy = ay_ad - ey_ad
                    fx = dx * ce + dy * se
                    fy = -(dx * se - dy * ce)
                    fz = az_ad - ez_ad
                    # ego frame → sensor frame (subtract offset, rotate by -lidar_yaw)
                    lx, ly, lz, lyaw = lidar_offsets.get(channel, (0.0, 0.0, 1.8, 0.0))
                    sx = fx - lx
                    sy = fy - ly
                    sz = fz - lz
                    cly, sly = m.cos(lyaw), m.sin(lyaw)
                    fx = sx * cly + sy * sly
                    fy = -(sx * sly - sy * cly)
                    fyaw = a["rotation"]["yaw"] - eyaw - m.degrees(lyaw)
                else:
                    fx, fy = loc["x"], loc["y"]
                    fyaw = a["rotation"]["yaw"]
                yaw = m.radians(fyaw)
                hx, hy = bb["x"] / 2, bb["y"] / 2
                cr, sr = m.cos(yaw), m.sin(yaw)
                corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
                rmat = np.array([[cr, -sr], [sr, cr]])
                corners = (rmat @ corners.T).T
                corners[:, 0] += fx; corners[:, 1] += fy
                px_c = ((rng - corners[:, 0]) * pix_per_m).astype(int)
                py_c = ((rng - corners[:, 1]) * pix_per_m).astype(int)
                pts = [(int(py_c[i]), int(px_c[i])) for i in range(4)]
                color = COLORS.get(a.get("category", "vehicle"), (0, 255, 0))
                draw.polygon(pts, outline=color)
                draw.text((int(py_c[0]), int(px_c[0]) - 10),
                          f"{a['category']}_{a['actor_id']}", fill=color)
            pil_img.save(os.path.join(ann_viz_dir, f"{frame_str}.png"))


# ══════════════════════════════════════════════════════════════════════
# OCC projection viz
# ══════════════════════════════════════════════════════════════════════

def _occ_projection_viz(run_dir, layout=None, force_all=False):
    if not force_all and layout is not None:
        for s in layout.get("sensors", []):
            if s.get("modality") == "occupancy":
                if not s.get("projection_vis", True):
                    return
                break
    occ_files = sorted(glob.glob(os.path.join(run_dir, "OCC", "original", "*.npy")))
    if not occ_files:
        return
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.occ_projection import _run_virtual_view
    _run_virtual_view(run_dir, [os.path.basename(f) for f in occ_files], None, 2)


# ══════════════════════════════════════════════════════════════════════
# OCC BEV visualization
# ══════════════════════════════════════════════════════════════════════

def _occ_bev_visualization(run_dir):
    npy_files = sorted(glob.glob(os.path.join(run_dir, "OCC", "original", "*.npy")))
    if not npy_files:
        return
    viz_dir = os.path.join(run_dir, "OCC", "occ_viz")
    os.makedirs(viz_dir, exist_ok=True)
    grid0 = np.load(npy_files[0])
    if grid0.ndim != 3:
        return
    meta_path = os.path.join(run_dir, "OCC", "occ_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        pc = meta.get("pc_range", [-50, -50, -5, 50, 50, 3])
        x_min, x_max, y_min, y_max = pc[0], pc[3], pc[1], pc[4]
        res = meta.get("voxel_size", [0.5, 0.5, 0.5])[0]
    else:
        x_min, x_max = -50, 50; y_min, y_max = -50, 50; res = 0.5
    print(f"OCC/occ_viz: {len(npy_files)} files")
    for npy_path in tqdm(npy_files, desc="OCC BEV", leave=False):
        grid = np.load(npy_path)
        bev = np.max(grid, axis=2)
        bev_img = np.rot90(np.flipud(bev.T), k=1)
        rgb = np.zeros((bev_img.shape[0], bev_img.shape[1], 3), dtype=np.uint8)
        for cat, color in OCC_COLORS.items():
            rgb[bev_img == cat] = color
        scale = 6
        half = min(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
        r1 = max(0, int((x_max - half) / res))
        r2 = min(bev_img.shape[0], int((x_max + half) / res))
        c1 = max(0, int((-half - y_min) / res))
        c2 = min(bev_img.shape[1], int((half - y_min) / res))
        if r2 > r1 and c2 > c1:
            rgb = rgb[r1:r2, c1:c2]
        rgb_big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        h, w = rgb_big.shape[:2]
        cr, cc = h // 2, w // 2
        if 0 <= cr < h and 0 <= cc < w:
            rr, cc2 = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            rgb_big[(rr - cr) ** 2 + (cc2 - cc) ** 2 < (1 * scale) ** 2] = (0, 255, 0)
        fname = os.path.basename(npy_path).replace(".npy", ".png")
        Image.fromarray(rgb_big).save(os.path.join(viz_dir, fname))


def _filter_bev_viz(run_dir):
    occ_dir = os.path.join(run_dir, "LIDAR_FILTER", "original")
    if not os.path.isdir(occ_dir):
        return
    npy_files = sorted(glob.glob(os.path.join(occ_dir, "*.npy")))
    if not npy_files:
        return
    meta_path = os.path.join(run_dir, "LIDAR_FILTER", "filter_meta.json")
    x_min = x_max = y_min = y_max = res = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        pc = meta.get("pc_range", [-80, -80, -5, 80, 80, 10])
        x_min, x_max, y_min, y_max = pc[0], pc[3], pc[1], pc[4]
        res = meta.get("voxel_size", [0.5, 0.5, 0.5])[0]
    grid0 = np.load(npy_files[0])
    if grid0.ndim != 3:
        return
    if x_min is None:
        x_min, x_max = -80, 80; y_min, y_max = -80, 80; res = 0.5
    viz_dir = os.path.join(run_dir, "LIDAR_FILTER", "occ_viz")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"LIDAR_FILTER/occ_viz: {len(npy_files)} files")
    for npy_path in tqdm(npy_files, desc="OCC_FILTER BEV", leave=False):
        grid = np.load(npy_path)
        bev = np.max(grid, axis=2)
        bev_img = np.rot90(np.flipud(bev.T), k=1)
        rgb = np.zeros((bev_img.shape[0], bev_img.shape[1], 3), dtype=np.uint8)
        for cat, color in OCC_COLORS.items():
            rgb[bev_img == cat] = color
        scale = 6
        half = min(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
        r1 = max(0, int((x_max - half) / res))
        r2 = min(bev_img.shape[0], int((x_max + half) / res))
        c1 = max(0, int((-half - y_min) / res))
        c2 = min(bev_img.shape[1], int((half - y_min) / res))
        if r2 > r1 and c2 > c1:
            rgb = rgb[r1:r2, c1:c2]
        rgb_big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        h, w = rgb_big.shape[:2]
        cr, cc = h // 2, w // 2
        if 0 <= cr < h and 0 <= cc < w:
            rr, cc2 = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            rgb_big[(rr - cr) ** 2 + (cc2 - cc) ** 2 < (1 * scale) ** 2] = (0, 255, 0)
        fname = os.path.basename(npy_path).replace(".npy", ".png")
        Image.fromarray(rgb_big).save(os.path.join(viz_dir, fname))


# ══════════════════════════════════════════════════════════════════════
# Trajectory visualization
# ══════════════════════════════════════════════════════════════════════

def _load_ego_poses(run_dir):
    poses = {}
    path = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if not os.path.exists(path):
        return poses
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            try:
                p = {"x": float(parts[1]), "y": -float(parts[2]), "z": float(parts[3]),
                     "roll": -float(parts[4]), "pitch": float(parts[5]), "yaw": -float(parts[6])}
                poses[int(parts[0])] = p
            except (ValueError, IndexError):
                pass
    return poses


def _trajectory_visualization(run_dir, layout, force_all=False):
    if not force_all and not layout.get("trajectory_vis", True):
        return
    poses = _load_ego_poses(run_dir)
    if not poses:
        return
    frames = sorted(poses.keys())
    if len(frames) < 1:
        return
    xs = [p["x"] for p in poses.values()]
    ys = [p["y"] for p in poses.values()]
    margin = 15.0
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin
    span = max(x_max - x_min, y_max - y_min, 1.0)
    rng = span / 2 + margin
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    res = 0.2
    size = int(2 * rng / res)
    viz_dir = os.path.join(run_dir, "TRAJ", "trajectory_viz")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"trajectory_viz: {len(frames)} frames")
    for i, frame in enumerate(tqdm(frames, desc="Trajectory viz", leave=False)):
        hist_x, hist_y = [], []
        for j in range(i + 1):
            p = poses[frames[j]]
            hist_x.append(p["x"]); hist_y.append(p["y"])
        img = np.zeros((size, size, 3), dtype=np.uint8)
        pix_per_m = size / (2 * rng)
        pts_pix = []
        for fx, fy in zip(hist_x, hist_y):
            px = int((fy - cy + rng) * pix_per_m)
            py = int(size - (fx - (cx - rng)) * pix_per_m)
            pts_pix.append((px, py))
        if len(pts_pix) >= 2:
            pil_tmp = Image.fromarray(img)
            draw_tmp = ImageDraw.Draw(pil_tmp)
            draw_tmp.line(pts_pix, fill=(50, 100, 200), width=2)
            img = np.array(pil_tmp)
        if pts_pix:
            px, py = pts_pix[-1]
            rr, cc = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            img[(rr - py) ** 2 + (cc - px) ** 2 < 9] = (0, 255, 0)
        scale = 4
        img_big = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        Image.fromarray(img_big).save(os.path.join(viz_dir, f"{frame:08d}.png"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/visualize.py output/<run_dir> [--all]")
        print("  --all  强制生成所有可视化，忽略 sensor_layout.yaml 中的开关")
        sys.exit(1)
    force_all = "--all" in sys.argv
    convert_run(sys.argv[1], force_all=force_all)
