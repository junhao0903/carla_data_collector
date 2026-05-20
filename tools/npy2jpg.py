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


def _load_sensor_layout(run_dir):
    """Load sensor layout YAML with vis flags, saved by collector."""
    path = os.path.join(run_dir, "sensor_layout.yaml")
    if os.path.exists(path):
        with open(path) as f:
            import yaml
            return yaml.safe_load(f)
    return {}

def _cam_vis_enabled(layout, channel, key, default=True):
    """Check per-camera vis flag in sensor layout."""
    for s in layout.get("sensors", []):
        if s.get("channel") == channel:
            return s.get(key, default)
    return default

def convert_run(run_dir, quality=95, force_all=False):
    layout = _load_sensor_layout(run_dir)
    _convert_orin(run_dir, quality)
    _depth_visualization(run_dir, layout, quality, force_all)
    _semantic_visualization(run_dir, layout, force_all)
    _generate_occ(run_dir, layout, force_all)
    _annotation_visualization(run_dir, layout, force_all)
    _trajectory_visualization(run_dir, layout, force_all)


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
    # Try new metadata format first (OCC/occ_metadata.json)
    new_meta = os.path.join(run_dir, "OCC", "occ_metadata.json")
    if os.path.exists(new_meta):
        with open(new_meta) as f:
            cfg = json.load(f)
        pc = cfg.get("pc_range", [-50, -50, -5, 50, 50, 3])
        vs = cfg.get("voxel_size", [0.5, 0.5, 0.5])
        return (pc[0], pc[3], pc[1], pc[4], pc[2], pc[5], vs[0])
    # Fall back to old grid_config.json
    occ_dir = os.path.join(run_dir, "OCC")
    cfg_path = os.path.join(occ_dir, "grid_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        return (cfg["x_min_m"], cfg["x_max_m"], cfg["y_min_m"], cfg["y_max_m"],
                cfg.get("z_min_m", -2), cfg.get("z_max_m", 4), cfg["resolution_m"])
    return (-20, 80, -40, 40, -2, 4, 0.5)


def _load_ego_poses(run_dir):
    poses = {}
    path = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
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
    hfov = m.radians(70)
    vfov = 2 * m.atan(m.tan(hfov / 2) * h / w)
    fx = w / (2 * m.tan(hfov / 2))
    fy = h / (2 * m.tan(vfov / 2))
    cx, cy = w / 2, h / 2

    # Camera → world rotation matrix
    cy = m.cos(cam_yaw); sy = m.sin(cam_yaw)
    cp = m.cos(cam_pitch); sp = m.sin(cam_pitch)
    cr = m.cos(cam_roll); sr = m.sin(cam_roll)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_cam = R_yaw @ R_pitch @ R_roll  # camera→world rotation

    # LiDAR data is already in ego frame: X=fwd, Y=left, Z=sensor-local+1.8=ego
    lx_ego = lidar_points[:, 0]
    ly_ego = lidar_points[:, 1]
    lz_ego = lidar_points[:, 2] + 1.8

    # Camera in ego: at (1.5, 0, 1.6), same orientation as ego
    # Direct ego→camera (no world round-trip)
    cam_X = lx_ego - 1.5   # forward (camera is 1.5m ahead of ego center)
    cam_Y = ly_ego          # left (camera is at y=0)
    cam_Z = lz_ego - 1.6   # up (camera is 1.6m above ego origin)
    pts_cam = np.stack([cam_X, cam_Y, cam_Z], axis=-1)

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


def _semantic_visualization(run_dir, layout, force_all=False):
    """Generate colorized semantic PNGs in semantic_viz/"""
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
        for png_path in png_files:
            tags = np.array(Image.open(png_path))  # (H, W) uint8
            rgb = np.zeros((tags.shape[0], tags.shape[1], 3), dtype=np.uint8)
            for tag, color in OCC_COLORS.items():
                rgb[tags == tag] = color
            fname = os.path.basename(png_path)
            Image.fromarray(rgb).save(os.path.join(viz_dir, fname))


def _annotation_visualization(run_dir, layout, force_all=False):
    """Draw 2D/3D bounding boxes on camera and OCC images."""
    _camera_annotation_viz(run_dir, layout, force_all)
    _lidar_annotation_viz(run_dir, layout, force_all)


def _camera_annotation_viz(run_dir, layout, force_all=False):
    import json
    COLORS = {"vehicle": (0, 255, 0), "pedestrian": (0, 0, 255),
         "static_car": (255, 200, 0), "static_truck": (255, 150, 0),
         "static_bus": (255, 100, 0), "static_train": (255, 50, 0),
         "static_motorcycle": (255, 200, 50), "static_bicycle": (255, 200, 100),
         "static_pedestrian": (200, 100, 255)}

    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        channel = os.path.basename(cam_dir)
        if not force_all and not _cam_vis_enabled(layout, channel, "annotation_vis", True):
            continue
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
                if x2 <= x1 or y2 <= y1:
                    continue
                color = COLORS.get(a.get("category", "vehicle"), (0, 255, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label = f"{a['category']}_{a['actor_id']}"
                draw.text((x1, y1 - 10), label, fill=color)
            img.save(os.path.join(viz_dir, fname))


def _lidar_annotation_viz(run_dir, layout, force_all=False):
    """Draw 3D LiDAR annotation bboxes on raw LiDAR point cloud BEV."""
    import json
    from PIL import ImageDraw
    if not force_all:
        for s in layout.get("sensors", []):
            if s.get("modality") in ("lidar", "lidar_semantic") and s.get("enabled", True):
                if not s.get("annotation_vis", True):
                    return
                break

    for lidar_channel in sorted(glob.glob(os.path.join(run_dir, "LIDAR_*"))):
        channel = os.path.basename(lidar_channel)
        lidar_dir = os.path.join(run_dir, channel, "original")
        lidar_ann_dir = os.path.join(run_dir, channel, "annotations")
        if not os.path.isdir(lidar_dir) or not os.path.isdir(lidar_ann_dir):
            continue

        COLORS = {"vehicle": (0, 255, 0), "pedestrian": (0, 0, 255),
             "static_car": (255, 200, 0), "static_truck": (255, 150, 0),
             "static_bus": (255, 100, 0), "static_motorcycle": (255, 200, 50),
             "static_bicycle": (255, 200, 100)}
        ann_viz_dir = os.path.join(run_dir, channel, "annotations_viz")
        os.makedirs(ann_viz_dir, exist_ok=True)

        lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.npy")))
        rng, scale, vmin, vmax = 60, 6, -1.5, 3.0  # range ±60m, 6x upscale, Z clip
        size = int(2 * rng / 0.1)  # 0.1m/pixel raw

        print(f"{channel}/annotations_viz: {len(lidar_files)} frames")
        for lpath in lidar_files:
            frame_str = os.path.basename(lpath).replace(".npy", "")
            ann_path = os.path.join(lidar_ann_dir, f"{frame_str}.json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path) as f:
                anns = json.load(f)
            if not anns:
                continue

            points = np.load(lpath)  # (N, 4) or (N, 6), X=fwd, Y=left, Z=up
            lx, ly, lz = points[:,0], points[:,1], points[:,2] + 1.8  # sensor→ego Z shift

            # Clip Z and XY range
            z_valid = (lz > vmin) & (lz < vmax)
            xy_valid = (np.abs(lx) < rng) & (np.abs(ly) < rng)
            valid = z_valid & xy_valid
            lx, ly, lz = lx[valid], ly[valid], lz[valid]
            if len(lx) == 0:
                continue

            # Create BEV image (intensity from Z)
            img = np.zeros((size, size), dtype=np.uint8)
            px = ((rng - lx) / (2*rng) * size).astype(int)  # X=fwd → up
            py = ((rng - ly) / (2*rng) * size).astype(int)    # Y=left → left
            px = np.clip(px, 0, size-1); py = np.clip(py, 0, size-1)
            # Height color: low=dark, high=bright
            lz_clip = np.clip(lz, vmin, vmax)
            intensity = ((lz_clip - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            np.maximum.at(img, (px, py), intensity)

            # Colorize and upscale
            img_color = np.stack([img, img, img], axis=-1)
            img_big = np.repeat(np.repeat(img_color, scale, axis=0), scale, axis=1)
            h_img, w_img = img_big.shape[:2]

            # Ego marker at center
            cy, cx = h_img//2, w_img//2
            rr, cc = np.meshgrid(np.arange(h_img), np.arange(w_img), indexing='ij')
            img_big[(rr-cy)**2 + (cc-cx)**2 < (1*scale)**2] = (0, 255, 0)

            # Draw 3D bboxes
            pil_img = Image.fromarray(img_big)
            draw = ImageDraw.Draw(pil_img)
            pix_per_m = scale * size / (2*rng)  # pixels per meter
            for a in anns:
                loc = a["location"]; bb = a["bbox_3d"]
                if bb["x"] == 0 or bb["y"] == 0:
                    continue
                fx, fy = loc["x"], loc["y"]
                yaw = m.radians(a["rotation"]["yaw"])
                hx, hy = bb["x"]/2, bb["y"]/2
                cr, sr = m.cos(yaw), m.sin(yaw)
                corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
                rot = np.array([[cr, -sr], [sr, cr]])
                corners = (rot @ corners.T).T
                corners[:,0] += fx; corners[:,1] += fy
                # Ego → pixel
                px_c = ((rng - corners[:,0]) * pix_per_m).astype(int)
                py_c = ((rng - corners[:,1]) * pix_per_m).astype(int)
                pts = [(int(py_c[i]), int(px_c[i])) for i in range(4)]
                color = COLORS.get(a.get("category", "vehicle"), (0, 255, 0))
                draw.polygon(pts, outline=color)
                draw.text((int(py_c[0]), int(px_c[0])-10), f"{a['category']}_{a['actor_id']}", fill=color)

            pil_img.save(os.path.join(ann_viz_dir, f"{frame_str}.png"))


def _generate_gt_occ(run_dir, ann_dir, ego_csv, meta_path):
    """Generate OCC npy files from GT annotations + static_occ. Saves to OCC/original/."""
    import csv as _csv
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.occ_generator import build_frame_occ, PC_RANGE, VOXEL_SIZE, OCC_SHAPE

    with open(meta_path) as f:
        meta = json.load(f)
    pc_range = meta.get("pc_range", PC_RANGE)
    voxel_size = meta.get("voxel_size", VOXEL_SIZE)
    occ_shape = [int(round((pc_range[3]-pc_range[0])/voxel_size[0])),
                 int(round((pc_range[4]-pc_range[1])/voxel_size[1])),
                 int(round((pc_range[5]-pc_range[2])/voxel_size[2]))]

    static_dir = meta.get("static_occ_dir", "map")
    town = meta.get("map", "Town10HD")
    static_path = os.path.join(static_dir, f"{town}_static_occ.npy")
    if not os.path.exists(static_path):
        print(f"  Static OCC not found at {static_path}, skipping GT OCC generation")
        return
    static_occ = np.load(static_path)
    with open(static_path.replace(".npy", ".json")) as f:
        static_pc_range = json.load(f)["pc_range"]

    # Load ego poses (left-hand CSV → CARLA world)
    ego_frames = {}
    with open(ego_csv) as f:
        for row in _csv.DictReader(f):
            ego_frames[int(row["frame"])] = (
                float(row["x"]), -float(row["y_left"]), float(row["z"]),
                -float(row["roll_left"]), float(row["pitch"]), -float(row["yaw_left"]))

    # Load annotations keyed by frame
    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    anns_by_frame = {}
    for ap in ann_files:
        frame = int(os.path.basename(ap).replace(".json", ""))
        with open(ap) as f:
            anns_by_frame[frame] = json.load(f)

    out_dir = os.path.join(run_dir, "OCC", "original")
    os.makedirs(out_dir, exist_ok=True)

    try:
        import carla
    except ImportError:
        print("  carla module not available, skipping GT OCC generation")
        return

    count = 0
    for frame, (ex, ey, ez, eroll, epitch, eyaw) in ego_frames.items():
        anns = anns_by_frame.get(frame)
        if not anns:
            continue
        ego_tf = carla.Transform(
            carla.Location(x=ex, y=ey, z=ez),
            carla.Rotation(roll=eroll, pitch=epitch, yaw=eyaw))
        eyaw_rad = m.radians(eyaw)
        cy, sy = m.cos(eyaw_rad), m.sin(eyaw_rad)
        dynamic_list = []
        for a in anns:
            loc = a["location"]; rot = a["rotation"]; ext = a["extent"]
            # Ego frame (fwd=X, left=Y) → CARLA world (fwd=X, right=Y)
            fx, fy, fz = loc["x"], loc["y"], loc["z"]
            wx = ex + fx * cy + fy * sy
            wy = ey + fx * sy - fy * cy
            wz = ez + fz
            # Left-hand rotation → CARLA world rotation
            wroll = -rot["roll"]
            wpitch = rot["pitch"]
            wyaw = eyaw - rot["yaw"]
            actor_tf = carla.Transform(
                carla.Location(x=wx, y=wy, z=wz),
                carla.Rotation(roll=wroll, pitch=wpitch, yaw=wyaw))
            actor_ext = carla.Vector3D(x=ext["x"], y=ext["y"], z=ext["z"])
            dynamic_list.append((actor_tf, actor_ext, a.get("type_id", "vehicle.car")))
        occ = build_frame_occ(static_occ, static_pc_range, ego_tf, dynamic_list,
                              pc_range, voxel_size, occ_shape)
        np.save(os.path.join(out_dir, f"{frame:08d}.npy"), occ)
        count += 1

    print(f"OCC generated from GT: {count} frames → {out_dir}")


def _generate_occ(run_dir, layout=None, force_all=False):
    """Generate OCC npy + BEV visualization.

    1. If OCC/original/*.npy exists → visualize directly
    2. Elif OCC/annotations/ exists → auto-generate npy from GT annotations + static_occ, then visualize
    3. Else → fall back to old LiDAR pipeline
    """
    if not force_all and layout is not None:
        occ_enabled = False
        for s in layout.get("sensors", []):
            if s.get("modality") == "occupancy":
                occ_enabled = s.get("enabled", True)
                if not s.get("occ_vis", True):
                    return
                break
        if not occ_enabled:
            return
    occ_original = os.path.join(run_dir, "OCC", "original")
    npy_files = sorted(glob.glob(os.path.join(occ_original, "*.npy"))) if os.path.isdir(occ_original) else []

    if npy_files:
        _occ_bev_visualization(run_dir, npy_files)
        return

    # Auto-generate OCC from GT annotations if available
    ann_dir = os.path.join(run_dir, "OCC", "annotations")
    if os.path.isdir(ann_dir) and os.listdir(ann_dir):
        ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
        meta_path = os.path.join(run_dir, "OCC", "occ_metadata.json")
        if os.path.exists(ego_csv) and os.path.exists(meta_path):
            _generate_gt_occ(run_dir, ann_dir, ego_csv, meta_path)
            npy_files = sorted(glob.glob(os.path.join(occ_original, "*.npy")))
            if npy_files:
                _occ_bev_visualization(run_dir, npy_files)
                return

    # Fallback: old LiDAR-based pipeline (backward compatible)
    x_min, x_max, y_min, y_max, z_min, z_max, res = _load_grid_params(run_dir)
    occ_dir = os.path.join(run_dir, "OCC")
    os.makedirs(occ_dir, exist_ok=True)

    lidar_dir = os.path.join(run_dir, "LIDAR_TOP", "original")
    if not os.path.isdir(lidar_dir):
        return

    ego_poses = _load_ego_poses(run_dir)
    nx = int(round((x_max - x_min) / res))
    ny = int(round((y_max - y_min) / res))
    nz = int(round((z_max - z_min) / res))

    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.npy")))
    print(f"OCC from LiDAR: {len(lidar_files)} frames → {occ_dir}")
    for lpath in lidar_files:
        frame_str = os.path.basename(lpath).replace(".npy", "")
        frame = int(frame_str)
        occ_path = os.path.join(occ_dir, f"{frame_str}.npy")
        points = np.load(lpath)
        ex = points[:, 0]; ey = points[:, 1]; ez = points[:, 2] + 1.8
        ix = np.floor((ex-x_min)/res).astype(np.int32)
        iy = np.floor((ey-y_min)/res).astype(np.int32)
        iz = np.floor((ez-z_min)/res).astype(np.int32)
        valid = (ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)&(iz>=0)&(iz<nz)
        if valid.sum() == 0:
            continue
        ix, iy, iz = ix[valid], iy[valid], iz[valid]
        tags = _label_lidar_with_camera(run_dir, points, frame, ego_poses)
        tags = tags[valid]
        occ_idx = ix * ny * nz + iy * nz + iz
        _, ui = np.unique(occ_idx, return_index=True)
        ix, iy, iz, tags = ix[ui], iy[ui], iz[ui], tags[ui]
        grid = np.zeros((nz, ny, nx), dtype=np.uint8)
        grid[iz, iy, ix] = tags
        np.save(occ_path, grid)

    # Old actor overlay (kept for backward compat)
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
            for a in anns:
                cat = actor_map.get(a.get("category"), 21)
                loc, rot, bb = a["location"], a["rotation"], a["bbox_3d"]
                cx, cy_e, cz_e = loc["x"], loc["y"], loc["z"]
                rel_yaw = m.radians(rot["yaw"])
                cr, sr = m.cos(rel_yaw), m.sin(rel_yaw)
                hx, hy, hz = bb["x"]/2, bb["y"]/2, bb["z"]/2
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

    _occ_bev_visualization(run_dir, sorted(glob.glob(os.path.join(occ_dir, "*.npy"))), x_min, x_max, y_min, y_max, res)


def _occ_bev_visualization(run_dir, npy_files, x_min=None, x_max=None, y_min=None, y_max=None, res=None):
    """BEV visualization of OCC npy files. Handles (X,Y,Z) grid format.

    Grid layout: axis0=X (forward), axis1=Y (left), axis2=Z (up).
    BEV: max over Z → (X,Y), rotated 90° CCW so forward=left, left=down.
    """
    if not npy_files:
        return

    viz_dir = os.path.join(run_dir, "OCC", "occ_viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Auto-detect grid params from first npy + metadata
    grid0 = np.load(npy_files[0])
    if grid0.ndim != 3:
        return
    nx, ny = grid0.shape[0], grid0.shape[1]

    if x_min is None:
        meta_path = os.path.join(run_dir, "OCC", "occ_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            pc = meta.get("pc_range", [-50, -50, -5, 50, 50, 3])
            x_min, x_max, y_min, y_max = pc[0], pc[3], pc[1], pc[4]
            res = meta.get("voxel_size", [0.5, 0.5, 0.5])[0]
        else:
            x_min, x_max = -50, 50
            y_min, y_max = -50, 50
            res = 0.5

    print(f"OCC/occ_viz: {len(npy_files)} files")

    for npy_path in npy_files:
        grid = np.load(npy_path)
        # BEV: max over Z (axis=2), result (X, Y)
        bev = np.max(grid, axis=2)  # (nx, ny)
        # Transpose + flip → forward=top, left=left, then rotate 90° CCW
        bev_img = np.rot90(np.flipud(bev.T), k=1)

        rgb = np.zeros((bev_img.shape[0], bev_img.shape[1], 3), dtype=np.uint8)
        for cat, color in OCC_COLORS.items():
            rgb[bev_img == cat] = color

        scale = 6
        # Center crop around ego (ego at center of grid)
        half = min(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
        r1 = max(0, int((x_max - half) / res))
        r2 = min(bev_img.shape[0], int((x_max + half) / res))
        c1 = max(0, int((-half - y_min) / res))
        c2 = min(bev_img.shape[1], int((half - y_min) / res))
        if r2 > r1 and c2 > c1:
            rgb = rgb[r1:r2, c1:c2]

        rgb_big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        # Ego marker at center
        h, w = rgb_big.shape[:2]
        cr, cc = h // 2, w // 2
        if 0 <= cr < h and 0 <= cc < w:
            rr, cc2 = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            mask = (rr - cr) ** 2 + (cc2 - cc) ** 2 < (1 * scale) ** 2
            rgb_big[mask] = (0, 255, 0)

        fname = os.path.basename(npy_path).replace(".npy", ".png")
        Image.fromarray(rgb_big).save(os.path.join(viz_dir, fname))


def _trajectory_visualization(run_dir, layout, force_all=False):
    """Draw ego trajectory BEV, accumulating over frames. 90° CCW: X=fwd→up, Y=left→left."""
    if not force_all and not layout.get("trajectory_vis", True):
        return
    poses = _load_ego_poses(run_dir)
    if not poses:
        return
    frames = sorted(poses.keys())
    if len(frames) < 1:
        return

    # Determine range from trajectory extent
    xs = [p["x"] for p in poses.values()]
    ys = [p["y"] for p in poses.values()]
    margin = 15.0
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin
    span_x, span_y = x_max - x_min, y_max - y_min
    span = max(span_x, span_y, 1.0)

    rng = span / 2 + margin
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    res = 0.2  # m/pixel
    size = int(2 * rng / res)

    viz_dir = os.path.join(run_dir, "TRAJ", "trajectory_viz")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"trajectory_viz: {len(frames)} frames")

    for i, frame in enumerate(frames):
        hist_x, hist_y = [], []
        for j in range(i + 1):
            p = poses[frames[j]]
            hist_x.append(p["x"])
            hist_y.append(p["y"])

        img = np.zeros((size, size, 3), dtype=np.uint8)
        pix_per_m = size / (2 * rng)

        from PIL import ImageDraw as _ImageDraw
        pil_tmp = Image.fromarray(img)
        draw_tmp = _ImageDraw.Draw(pil_tmp)
        pts_pix = []
        for fx, fy in zip(hist_x, hist_y):
            # 90° CCW: forward(X)→up, left(Y)→left
            # px = (cy+rng - fy) normalized, py = size - (fx - (cx-rng)) normalized
            px = int(((cy + rng) - fy) * pix_per_m)   # Y=left → image X=left
            py = int(size - (fx - (cx - rng)) * pix_per_m)  # X=fwd → image Y=up
            pts_pix.append((px, py))
        if len(pts_pix) >= 2:
            draw_tmp.line(pts_pix, fill=(50, 100, 200), width=2)
        img = np.array(pil_tmp)

        if pts_pix:
            px, py = pts_pix[-1]
            rr, cc = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            dist = np.sqrt((rr - py)**2 + (cc - px)**2)
            img[dist < 3] = (0, 255, 0)

        # Scale up for visibility
        scale = 4
        img_big = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        fname = f"{frame:08d}.png"
        Image.fromarray(img_big).save(os.path.join(viz_dir, fname))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/npy2jpg.py output/<run_dir> [--all]")
        print("  --all  强制生成所有可视化，忽略 sensor_layout.yaml 中的开关")
        sys.exit(1)
    force_all = "--all" in sys.argv
    convert_run(sys.argv[1], force_all=force_all)
