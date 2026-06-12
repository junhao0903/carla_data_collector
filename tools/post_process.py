#!/usr/bin/env python3
"""Post-processing: convert raw sensor data, generate OCC, filter annotations.

Usage: python tools/post_process.py <run_dir>
"""
import sys, os, json, math as m, glob, csv as _csv
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _load_ego_trajectory(run_dir):
    """Load ego trajectory, convert from right-hand world to CARLA world."""
    ego = {}
    path = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if not os.path.exists(path):
        return ego
    with open(path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            ego[int(row["frame"])] = (
                float(row["x"]), float(row["y_left"]), float(row["z"]),
                float(row["roll_left"]), float(row["pitch"]), float(row["yaw_left"]))
    return ego


def _load_filter_config():
    for search in ["config/filter/default.yaml"]:
        if os.path.exists(search):
            import yaml
            with open(search) as f:
                return yaml.safe_load(f) or {}
    return {}


def _load_sensor_layout(run_dir):
    path = os.path.join(run_dir, "sensor_layout.yaml")
    if os.path.exists(path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def _load_grid_params(run_dir):
    meta_path = os.path.join(run_dir, "OCC", "occ_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            cfg = json.load(f)
        pc = cfg.get("pc_range", [-50, -50, -5, 50, 50, 3])
        vs = cfg.get("voxel_size", [0.5, 0.5, 0.5])
        return (pc[0], pc[3], pc[1], pc[4], pc[2], pc[5], vs[0])
    return (-20, 80, -40, 40, -2, 4, 0.5)



# ══════════════════════════════════════════════════════════════════════
# Semantic: raw BGRA .npy → tag PNG
# ══════════════════════════════════════════════════════════════════════

def convert_semantic(run_dir):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        src_dir = os.path.join(cam_dir, "semantic")
        if not os.path.isdir(src_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
        if not npy_files:
            continue
        channel = os.path.basename(cam_dir)
        print(f"{channel}/semantic: converting {len(npy_files)} .npy → .png")
        for npy_path in tqdm(npy_files, desc=f"{channel} semantic", leave=True):
            png_path = npy_path.replace(".npy", ".png")
            if os.path.exists(png_path):
                continue
            raw = np.load(npy_path)
            from src.sensors import decode_semantic
            tags = decode_semantic(raw)
            Image.fromarray(tags, mode="L").save(png_path)


# ══════════════════════════════════════════════════════════════════════
# Depth: raw BGRA .npy → decoded depth .npy (meters, float32)
# ══════════════════════════════════════════════════════════════════════

def convert_depth(run_dir):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        src_dir = os.path.join(cam_dir, "depth")
        if not os.path.isdir(src_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
        if not npy_files:
            continue
        sample = np.load(npy_files[0])
        if sample.ndim != 3 or sample.shape[2] != 4:
            continue
        channel = os.path.basename(cam_dir)
        print(f"{channel}/depth: decoding {len(npy_files)} files")
        for npy_path in tqdm(npy_files, desc=f"{channel} depth", leave=True):
            raw = np.load(npy_path)
            from src.sensors import decode_depth
            depth = decode_depth(raw)
            np.save(npy_path, depth)


# ══════════════════════════════════════════════════════════════════════
# Original: raw BGRA .npy → RGB .jpg
# ══════════════════════════════════════════════════════════════════════

def convert_orin(run_dir, quality=95):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        sub_dir = os.path.join(cam_dir, "original")
        if not os.path.isdir(sub_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(sub_dir, "*.npy")))
        if not npy_files:
            continue
        channel = os.path.basename(cam_dir)
        print(f"{cam_dir}/original: {len(npy_files)} files -> jpg")
        for npy_path in tqdm(npy_files, desc=f"{channel} original", leave=True):
            arr = np.load(npy_path)
            if arr.ndim == 3 and arr.shape[2] == 4:
                jpg_path = npy_path.replace(".npy", ".jpg")
                Image.fromarray(arr[:, :, [2, 1, 0]]).save(jpg_path, quality=quality)
                os.remove(npy_path)


# ══════════════════════════════════════════════════════════════════════
# OCC generation
# ══════════════════════════════════════════════════════════════════════

def generate_gt_occ(run_dir, ann_dir, ego_csv, meta_path):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.occ_generator import build_frame_occ

    with open(meta_path) as f:
        meta = json.load(f)
    pc_range = meta.get("pc_range", [-50, -50, -5, 50, 50, 3])
    voxel_size = meta.get("voxel_size", [0.5, 0.5, 0.5])
    occ_shape = [int(round((pc_range[3] - pc_range[0]) / voxel_size[0])),
                 int(round((pc_range[4] - pc_range[1]) / voxel_size[1])),
                 int(round((pc_range[5] - pc_range[2]) / voxel_size[2]))]
    static_dir = meta.get("static_occ_dir", "map")
    town = meta.get("map", "Town10HD")
    static_path = os.path.join(static_dir, f"{town}_static_occ.npy")
    if not os.path.exists(static_path):
        print(f"  Static OCC not found at {static_path}, skipping OCC generation")
        return
    static_occ = np.load(static_path)
    with open(static_path.replace(".npy", ".json")) as f:
        static_pc_range = json.load(f)["pc_range"]

    ego_frames = {}
    with open(ego_csv) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            ego_frames[int(row["frame"])] = (
                float(row["x"]), float(row["y_left"]), float(row["z"]),
                float(row["roll_left"]), float(row["pitch"]), float(row["yaw_left"]))

    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    anns_by_frame = {}
    for ap in ann_files:
        frame = int(os.path.basename(ap).replace(".json", ""))
        with open(ap) as f:
            anns_by_frame[frame] = json.load(f)

    try:
        import carla
    except ImportError:
        print("carla module not available, skipping OCC generation")
        return

    out_dir = os.path.join(run_dir, "OCC", "original")
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for frame, (ex, ey, ez, eroll, epitch, eyaw) in tqdm(
            [(f, ego_frames[f]) for f in sorted(ego_frames) if f in anns_by_frame],
            desc="GT OCC", leave=True):
        anns = anns_by_frame[frame]
        ego_tf = carla.Transform(
            carla.Location(x=ex, y=ey, z=ez),
            carla.Rotation(roll=eroll, pitch=epitch, yaw=eyaw))
        dynamic_list = []
        for a in anns:
            loc = a["location"]; rot = a["rotation"]; sz = a["bbox_3d"]
            # AD coords (Y=left) → CARLA world (Y=right)
            actor_tf = carla.Transform(
                carla.Location(x=loc["x"], y=-loc["y"], z=loc["z"]),
                carla.Rotation(roll=-rot["roll"], pitch=rot["pitch"], yaw=-rot["yaw"]))
            actor_ext = carla.Vector3D(x=sz["x"] / 2, y=sz["y"] / 2, z=sz["z"] / 2)
            dynamic_list.append((actor_tf, actor_ext, a.get("type_id", "vehicle.car")))
        occ = build_frame_occ(static_occ, static_pc_range, ego_tf, dynamic_list,
                              pc_range, voxel_size, occ_shape)
        np.save(os.path.join(out_dir, f"{frame:08d}.npy"), occ)
        count += 1
    print(f"  OCC generated: {count} frames → {out_dir}")


def generate_occ(run_dir, layout=None):
    # Check if OCC already generated
    occ_original = os.path.join(run_dir, "OCC", "original")
    npy_files = sorted(glob.glob(os.path.join(occ_original, "*.npy"))) if os.path.isdir(occ_original) else []
    if npy_files:
        return

    # Auto-generate from dynamic actor annotations
    ann_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if os.path.isdir(ann_dir) and os.listdir(ann_dir):
        ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
        meta_path = os.path.join(run_dir, "OCC", "occ_metadata.json")
        if os.path.exists(ego_csv) and os.path.exists(meta_path):
            generate_gt_occ(run_dir, ann_dir, ego_csv, meta_path)
            return



# ══════════════════════════════════════════════════════════════════════
# LiDAR point-count annotation filter
# ══════════════════════════════════════════════════════════════════════

def remap_static_ids(run_dir):
    """Assign unique positive IDs to static bboxes in static_bboxes.json."""
    path = os.path.join(run_dir, "ANNO", "static_bboxes.json")
    if not os.path.exists(path):
        return

    # Find max dynamic actor ID across all annotation files
    max_dyn = 0
    ann_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if os.path.isdir(ann_dir):
        for fname in os.listdir(ann_dir):
            with open(os.path.join(ann_dir, fname)) as f:
                for a in json.load(f):
                    aid = a.get("actor_id", 0)
                    if aid > max_dyn: max_dyn = aid

    with open(path) as f:
        bboxes = json.load(f)

    offset = max_dyn + 1
    for i, bb in enumerate(bboxes):
        bb["actor_id"] = offset + i

    with open(path, "w") as f:
        json.dump(bboxes, f)
    print(f"  Static ID remap: {len(bboxes)} bboxes → [{offset}, {offset + len(bboxes) - 1}]")


def align_frames(run_dir):
    """Remove files and CSV rows outside the common frame range across all data dirs."""
    all_dirs = []
    for d in sorted(glob.glob(os.path.join(run_dir, "CAM_*", ""))):
        for sub in ["original", "depth", "semantic"]:
            sd = os.path.join(d, sub)
            if os.path.isdir(sd): all_dirs.append(sd)
    for d in sorted(glob.glob(os.path.join(run_dir, "LIDAR_*", ""))):
        for sub in ["original"]:
            sd = os.path.join(d, sub)
            if os.path.isdir(sd) and sd not in all_dirs: all_dirs.append(sd)
    da_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if os.path.isdir(da_dir) and da_dir not in all_dirs: all_dirs.append(da_dir)
    for d in [os.path.join(run_dir, "OCC", "original"),
              os.path.join(run_dir, "LIDAR_FILTER", "annotations")]:
        if os.path.isdir(d) and d not in all_dirs: all_dirs.append(d)

    frame_sets = []
    for ad in all_dirs:
        frames = {int(f.replace('.json','').replace('.npy','').replace('.jpg','').replace('.png',''))
                  for f in os.listdir(ad)}
        if frames: frame_sets.append((ad, frames))
    if len(frame_sets) < 2:
        return
    common = set(frame_sets[0][1])
    for _, fs in frame_sets[1:]: common &= fs
    trimmed = 0
    for ad, frames in frame_sets:
        for frm in frames - common:
            for ext in ['.json', '.npy', '.jpg', '.png']:
                fp = os.path.join(ad, f'{frm:08d}{ext}')
                if os.path.exists(fp):
                    os.remove(fp); trimmed += 1
    print(f"  Frame alignment: range [{min(common):08d}-{max(common):08d}], removed {trimmed} files")

    for csv_path in [os.path.join(run_dir, "GNSS", "data.csv"),
                     os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")]:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            lines = f.readlines()
        kept = [lines[0]]
        for line in lines[1:]:
            try:
                if int(line.split(",")[0]) in common:
                    kept.append(line)
            except ValueError:
                kept.append(line)
        if len(kept) < len(lines):
            with open(csv_path, "w") as f:
                f.writelines(kept)
            print(f"  CSV trimmed: {os.path.relpath(csv_path, run_dir)} ({len(lines) - len(kept)} rows)")


def _to_sensor_local(ax_ad, ay_ad, az_ad, aroll_ad, apitch_ad, ayaw_ad, ego, lidar_offsets):
    """Convert global AD annotation to sensor-local (X=fwd, Y=left, Z=up)."""
    ex_ad, ey_ad, ez_ad, eroll, epitch, eyaw = ego
    lx_off, ly_off, lz_off, lroll_off, lpitch_off, lyaw_off = lidar_offsets
    eyaw_r = m.radians(eyaw); ce, se = m.cos(eyaw_r), m.sin(eyaw_r)
    ep_r = m.radians(epitch); cp, sp = m.cos(ep_r), m.sin(ep_r)
    er_r = m.radians(eroll); cr, sr = m.cos(er_r), m.sin(er_r)
    dx = ax_ad - ex_ad; dy = ay_ad - ey_ad; dz = az_ad - ez_ad
    x1 = dx * ce + dy * se; y1 = -dx * se + dy * ce; z1 = dz
    x2 = x1 * cp - z1 * sp; y2 = y1; z2 = x1 * sp + z1 * cp
    fx = x2; fy = y2 * cr + z2 * sr; fz = -y2 * sr + z2 * cr
    sx = fx - lx_off; sy = fy - ly_off; sz = fz - lz_off
    cly, sly = m.cos(lyaw_off), m.sin(lyaw_off)
    clp, slp = m.cos(lpitch_off), m.sin(lpitch_off)
    clr, slr = m.cos(lroll_off), m.sin(lroll_off)
    xs1 = sx * cly + sy * sly; ys1 = -sx * sly + sy * cly; zs1 = sz
    xs2 = xs1 * clp - zs1 * slp; ys2 = ys1; zs2 = xs1 * slp + zs1 * clp
    xs3 = xs2; ys3 = ys2 * clr + zs2 * slr; zs3 = -ys2 * slr + zs2 * clr
    return (xs3, ys3, zs3,
            aroll_ad - eroll - m.degrees(lroll_off),
            apitch_ad - epitch - m.degrees(lpitch_off),
            ayaw_ad - eyaw - m.degrees(lyaw_off))


def _count_points(pts, sx, sy, sz, ayaw_s, hx, hy, hz):
    dx = pts[:, 0] - sx; dy = pts[:, 1] - sy; dz = pts[:, 2] - sz
    yaw = m.radians(ayaw_s); cr, sr = m.cos(yaw), m.sin(yaw)
    lx = dx * cr + dy * sr; ly = -dx * sr + dy * cr
    return int(((np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(dz) <= hz)).sum())


def overall_filter_annotations(run_dir):
    """LiDAR point-count + temporal filtering."""
    filter_cfg = _load_filter_config()
    min_pts = filter_cfg.get("min_points", 5)
    temporal_s = filter_cfg.get("temporal_window", 0.5)
    fps = filter_cfg.get("rate_hz", 20)
    temporal_frames = max(1, int(temporal_s * fps))

    ann_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if not os.path.isdir(ann_dir):
        print("LiDAR filter: no ANNO/dynamic_actors, skipping")
        return
    lidar_dir = os.path.join(run_dir, "LIDAR_FILTER", "original")
    if not os.path.isdir(lidar_dir):
        print("LiDAR filter: no filter LiDAR data, skipping")
        return
    out_dir = os.path.join(run_dir, "LIDAR_FILTER", "annotations")
    os.makedirs(out_dir, exist_ok=True)
    valid_dir = os.path.join(run_dir, "ANNO", "valid")
    os.makedirs(valid_dir, exist_ok=True)

    # Load ego poses in AD coords (X=fwd, Y=left) and LiDAR offset
    ego_ad = {}
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if os.path.exists(ego_csv):
        with open(ego_csv) as f:
            for row in _csv.DictReader(f):
                ego_ad[int(row["frame"])] = (
                    float(row["x"]), float(row["y_left"]), float(row["z"]),
                    float(row["roll_left"]), float(row["pitch"]), float(row["yaw_left"]))
    lidar_tf = filter_cfg.get("transform", {})
    lidar_offsets = (
        lidar_tf.get("x", 0.0),
        lidar_tf.get("y", 0.0),
        lidar_tf.get("z", 1.8),
        m.radians(lidar_tf.get("roll", 0.0)),
        m.radians(lidar_tf.get("pitch", 0.0)),
        m.radians(lidar_tf.get("yaw", 0.0)),
    )

    # Load static bboxes (global AD coords)
    static_bboxes = []
    static_path = os.path.join(run_dir, "ANNO", "static_bboxes.json")
    if os.path.exists(static_path):
        with open(static_path) as f:
            static_bboxes = json.load(f)

    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if not ann_files:
        return

    def _category(type_id):
        if str(type_id).startswith("vehicle."): return "vehicle"
        if str(type_id).startswith("walker.pedestrian."): return "pedestrian"
        return "vehicle"

    occ_filtered_count = 0
    recent_ids = []
    print(f"LiDAR filter: {len(ann_files)} frames (min_pts={min_pts}, temporal={temporal_s}s)")
    for ann_path in tqdm(ann_files, desc="LiDAR filter", leave=True):
        frame = int(os.path.basename(ann_path).replace(".json", ""))
        lidar_path = os.path.join(lidar_dir, f"{frame:08d}.npy")
        if not os.path.exists(lidar_path):
            continue
        pts = np.load(lidar_path)
        with open(ann_path) as f:
            anns = json.load(f)

        ego = ego_ad.get(frame)
        temporal_keep = set()
        for s in recent_ids:
            temporal_keep |= s

        filtered = []
        kept_this_frame = set()

        def _filter_actors(actor_list, apply_temporal=True):
            nonlocal occ_filtered_count
            for a in actor_list:
                bb = a.get("bbox_3d", {})
                hx = max(bb.get("x", 2.0) / 2, 1.0)
                hy = max(bb.get("y", 2.0) / 2, 0.5)
                hz = max(bb.get("z", 2.0) / 2, 0.5)
                if ego is not None:
                    rot = a.get("rotation", {})
                    sx, sy, sz, sroll, spitch, syaw = _to_sensor_local(
                        a["location"]["x"], a["location"]["y"], a["location"]["z"],
                        rot.get("roll", 0), rot.get("pitch", 0), rot.get("yaw", 0),
                        ego, lidar_offsets)
                else:
                    sx, sy, sz = a["location"]["x"], a["location"]["y"], a["location"]["z"]
                    sroll = a.get("rotation", {}).get("roll", 0)
                    spitch = a.get("rotation", {}).get("pitch", 0)
                    syaw = a.get("rotation", {}).get("yaw", 0)
                n = _count_points(pts, sx, sy, sz, syaw, hx, hy, hz)
                aid = a.get("actor_id", 0)
                if n < min_pts and (not apply_temporal or aid not in temporal_keep):
                    occ_filtered_count += 1
                    continue
                a["category"] = _category(a.get("type_id", ""))
                filtered.append(a)
                kept_this_frame.add(aid)

        _filter_actors(anns)
        if ego is not None:
            _filter_actors(static_bboxes, apply_temporal=True)

        # Save global AD coords to ANNO/valid/
        valid_path = os.path.join(valid_dir, f"{frame:08d}.json")
        with open(valid_path, "w") as f:
            json.dump(filtered, f)

        # Convert to sensor-local AD for LIDAR_FILTER/annotations/ (copy, don't mutate)
        out_path = os.path.join(out_dir, f"{frame:08d}.json")
        if ego is not None:
            sensor_local = []
            for a in filtered:
                rot = a.get("rotation", {})
                sx, sy, sz, sroll, spitch, syaw = _to_sensor_local(
                    a["location"]["x"], a["location"]["y"], a["location"]["z"],
                    rot.get("roll", 0), rot.get("pitch", 0), rot.get("yaw", 0),
                    ego, lidar_offsets)
                entry = dict(a)
                entry["location"] = {"x": sx, "y": sy, "z": sz}
                entry["rotation"] = {"roll": sroll, "pitch": spitch, "yaw": syaw}
                sensor_local.append(entry)
            with open(out_path, "w") as f:
                json.dump(sensor_local, f)
        else:
            with open(out_path, "w") as f:
                json.dump(filtered, f)
        recent_ids.append(kept_this_frame)
        if len(recent_ids) > temporal_frames:
            recent_ids.pop(0)

    if occ_filtered_count > 0:
        print(f"  LiDAR filtered: {occ_filtered_count} annotations")

    if not filter_cfg.get("output", False):
        import shutil
        d = os.path.join(run_dir, "LIDAR_FILTER")
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"  LIDAR_FILTER removed (output: false)")
        return

    print(f"  Done: {len(ann_files)} frames")
    print(f"  LIDAR_FILTER kept (output: true)")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def annotate_camera(run_dir):
    """Stable version based on _to_sensor_local (no SE3 rewrite)."""

    layout = _load_sensor_layout(run_dir)
    if not layout:
        return

    valid_dir = os.path.join(run_dir, "ANNO", "valid")
    if not os.path.isdir(valid_dir):
        return

    # =========================
    # ego poses
    # =========================
    ego_ad = {}
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")

    if os.path.exists(ego_csv):
        import csv as _csv
        with open(ego_csv) as f:
            for row in _csv.DictReader(f):
                ego_ad[int(row["frame"])] = (
                    float(row["x"]),
                    float(row["y_left"]),
                    float(row["z"]),
                    float(row["roll_left"]),
                    float(row["pitch"]),
                    float(row["yaw_left"]),
                )

    # =========================
    # intrinsics
    # =========================
    def compute_K(spec):
        out = spec.get("output", {})
        w = out.get("width", 1600)
        h = out.get("height", 900)

        if "fx" in out:
            return {
                "fx": out["fx"], "fy": out["fy"],
                "cx": out["cx"], "cy": out["cy"],
                "width": w, "height": h
            }

        hfov = m.radians(out.get("fov", 70))
        fx = w / (2 * m.tan(hfov / 2))
        vfov = 2 * m.atan(m.tan(hfov / 2) * h / w)
        fy = h / (2 * m.tan(vfov / 2))

        return {"fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2,
                "width": w, "height": h}

    # =========================
    # bbox corners
    # =========================
    def make_corners(bb):
        dx = max(bb.get("x", 2.0) / 2, 0.5)
        dy = max(bb.get("y", 2.0) / 2, 0.25)
        dz = max(bb.get("z", 2.0) / 2, 0.25)

        return np.array([
            [ dx,  dy,  dz],
            [ dx,  dy, -dz],
            [ dx, -dy,  dz],
            [ dx, -dy, -dz],
            [-dx,  dy,  dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [-dx, -dy, -dz],
        ])

    # =========================
    # projection
    # =========================
    def project(K, pts):
        fx, fy = K["fx"], K["fy"]
        cx, cy = K["cx"], K["cy"]

        X = pts[:, 0]
        Y = pts[:, 1]
        Z = pts[:, 2]

        valid = X > 0.1
        if valid.sum() == 0:
            return None

        u = fx * (-Y / X) + cx
        v = fy * (-Z / X) + cy

        return u, v, valid

    # =========================
    # main loop
    # =========================
    for s in layout.get("sensors", []):
        if not s.get("enabled", True):
            continue
        if s["modality"] != "camera_rgb":
            continue

        channel = s["channel"]
        K = compute_K(s)
        w, h = K["width"], K["height"]

        t = s.get("transform", {})

        cam_offsets = (
            t.get("x", 1.5),
            t.get("y", 0.0),
            t.get("z", 1.6),
            m.radians(t.get("roll", 0.0)),
            m.radians(t.get("pitch", 0.0)),
            m.radians(t.get("yaw", 0.0)),
        )

        ann_dir = os.path.join(run_dir, channel, "annotations")
        os.makedirs(ann_dir, exist_ok=True)

        for fname in sorted(os.listdir(valid_dir)):
            if not fname.endswith(".json"):
                continue

            frame = int(fname.replace(".json", ""))
            if frame not in ego_ad:
                continue

            ego = ego_ad[frame]

            with open(os.path.join(valid_dir, fname)) as f:
                anns = json.load(f)

            projected = []

            for a in anns:

                # =========================
                # 1. to sensor frame (ONLY place where transform happens)
                # =========================
                sx, sy, sz, sroll, spitch, syaw = _to_sensor_local(
                    a["location"]["x"],
                    a["location"]["y"],
                    a["location"]["z"],
                    a.get("rotation", {}).get("roll", 0.0),
                    a.get("rotation", {}).get("pitch", 0.0),
                    a.get("rotation", {}).get("yaw", 0.0),
                    ego,
                    cam_offsets
                )
                print("[SENSOR POS]", "sx:", sx, "sy:", sy, "sz:", sz)
                # debug point (VERY IMPORTANT)
                # print("sensor xyz:", sx, sy, sz)

                if sx <= 0.1:
                    continue

                # =========================
                # 2. bbox corners
                # =========================
                corners = make_corners(a.get("bbox_3d", {}))

                # ⚠️ ONLY yaw rotation (consistent with your pipeline)
                yaw = m.radians(syaw)
                cr, sr = m.cos(yaw), m.sin(yaw)

                R = np.array([
                    [cr, -sr, 0],
                    [sr,  cr, 0],
                    [0,   0,  1]
                ])

                corners = (R @ corners.T).T + np.array([sx, sy, sz])

                # =========================
                # 3. projection
                # =========================
                proj = project(K, corners)
                if proj is None:
                    continue

                u, v, valid = proj
                u = u[valid]
                v = v[valid]

                if len(u) < 2:
                    continue

                # =========================
                # 4. bbox
                # =========================
                xmin = int(np.clip(u.min(), 0, w - 1))
                xmax = int(np.clip(u.max(), 0, w - 1))
                ymin = int(np.clip(v.min(), 0, h - 1))
                ymax = int(np.clip(v.max(), 0, h - 1))

                if xmax <= xmin or ymax <= ymin:
                    continue

                entry = dict(a)
                entry["bbox_2d"] = [xmin, ymin, xmax, ymax]

                entry["location"] = {
                    "x": float(sx),
                    "y": float(sy),
                    "z": float(sz),
                }

                projected.append(entry)

            out_path = os.path.join(ann_dir, f"{frame:08d}.json")
            with open(out_path, "w") as f:
                json.dump(projected, f)

        print(f"[OK] {channel}")


def annotate_lidar(run_dir):
    """Project ANNO/valid annotations into each LiDAR sensor frame with FOV filter."""
    layout = _load_sensor_layout(run_dir)
    if not layout:
        return
    valid_dir = os.path.join(run_dir, "ANNO", "valid")
    if not os.path.isdir(valid_dir):
        return

    # Load ego poses in AD coords
    ego_ad = {}
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if os.path.exists(ego_csv):
        with open(ego_csv) as f:
            for row in _csv.DictReader(f):
                ego_ad[int(row["frame"])] = (
                    float(row["x"]), float(row["y_left"]), float(row["z"]),
                    float(row["roll_left"]), float(row["pitch"]), float(row["yaw_left"]))

    for s in layout.get("sensors", []):
        if not s.get("enabled", True):
            continue
        if s["modality"] not in ("lidar", "lidar_semantic"):
            continue
        channel = s["channel"]
        if channel == "LIDAR_FILTER":
            continue
        out = s.get("output", {})
        lidar_range = out.get("range", 100)
        upper_fov = m.radians(out.get("upper_fov", 10))
        lower_fov = m.radians(out.get("lower_fov", -30))
        h_fov = m.radians(out.get("horizontal_fov", 360))

        t = s.get("transform", {})
        lidar_offsets = (
            t.get("x", 0.0), t.get("y", 0.0), t.get("z", 1.8),
            m.radians(t.get("roll", 0.0)), m.radians(t.get("pitch", 0.0)),
            m.radians(t.get("yaw", 0.0)),
        )

        ann_dir = os.path.join(run_dir, channel, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        count = 0

        for fname in sorted(os.listdir(valid_dir)):
            frame = int(fname.replace(".json", ""))
            ego = ego_ad.get(frame)
            if ego is None:
                continue

            with open(os.path.join(valid_dir, fname)) as f:
                anns = json.load(f)

            projected = []
            for a in anns:
                rot = a.get("rotation", {})
                sx, sy, sz, sroll, spitch, syaw = _to_sensor_local(
                    a["location"]["x"], a["location"]["y"], a["location"]["z"],
                    rot.get("roll", 0), rot.get("pitch", 0), rot.get("yaw", 0),
                    ego, lidar_offsets)

                # Check if annotation center is within FOV
                d = m.sqrt(sx * sx + sy * sy + sz * sz)
                if d > lidar_range:
                    continue
                if d > 0.01:
                    elev = m.asin(sz / d)
                    if elev < lower_fov or elev > upper_fov:
                        continue
                if h_fov < m.radians(360) and d > 0.01:
                    azim = m.atan2(sy, sx)
                    if abs(azim) > h_fov / 2:
                        continue

                entry = dict(a)
                entry["location"] = {"x": sx, "y": sy, "z": sz}
                entry["rotation"] = {"roll": sroll, "pitch": spitch, "yaw": syaw}
                projected.append(entry)

            out_path = os.path.join(ann_dir, f"{frame:08d}.json")
            with open(out_path, "w") as f:
                json.dump(projected, f)
            count += 1

        print(f"  {channel}/annotations: {count} frames")


def simulate_async(run_dir):
    """Drop frames per sensor rate_hz to simulate async mode from sync data."""
    layout = _load_sensor_layout(run_dir)
    if not layout:
        return
    coll_cfg = layout.get("_collection", {})
    if coll_cfg.get("synchronous", True):
        return
    sync_fps = coll_cfg.get("fps", 20)

    for s in layout.get("sensors", []):
        if not s.get("enabled", True):
            continue
        rate = s.get("rate_hz")
        if not rate or rate >= sync_fps:
            continue
        step = sync_fps / rate  # e.g. 20/12 ≈ 1.67
        channel = s["channel"]
        for sub in ["original", "depth", "semantic"]:
            d = os.path.join(run_dir, channel, sub)
            if not os.path.isdir(d):
                continue
            all_files = sorted(glob.glob(os.path.join(d, "*")))
            kept = []
            acc = 0.0
            for f in all_files:
                acc += 1.0
                if acc >= step:
                    kept.append(f)
                    acc -= step
            for f in all_files:
                if f not in kept:
                    os.remove(f)
            if all_files:
                print(f"  {channel}/{sub}: kept {len(kept)}/{len(all_files)} (rate={rate}Hz)")


def dedup_static_bboxes(run_dir):
    """Remove static bboxes that overlap with dynamic actors in the first frame."""
    static_path = os.path.join(run_dir, "ANNO", "static_bboxes.json")
    if not os.path.exists(static_path):
        return
    dyn_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if not os.path.isdir(dyn_dir):
        return
    # Get first frame's dynamic actors
    dyn_files = sorted(os.listdir(dyn_dir))
    if not dyn_files:
        return
    with open(os.path.join(dyn_dir, dyn_files[0])) as f:
        dyn_actors = json.load(f)

    # Get ego position from first frame trajectory
    ego_pos = None
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if os.path.exists(ego_csv):
        with open(ego_csv) as f:
            for row in _csv.DictReader(f):
                ego_pos = (float(row["x"]), float(row["y_left"]))
                break

    with open(static_path) as f:
        static_bboxes = json.load(f)

    before = len(static_bboxes)
    static_bboxes = [sb for sb in static_bboxes
                     if not any(((sb["location"]["x"] - a["location"]["x"])**2 +
                                 (sb["location"]["y"] - a["location"]["y"])**2)**0.5 < 2.0
                                for a in dyn_actors)
                     and not (ego_pos is not None and
                              ((sb["location"]["x"] - ego_pos[0])**2 +
                               (sb["location"]["y"] - ego_pos[1])**2)**0.5 < 0.5)]
    after = len(static_bboxes)
    with open(static_path, "w") as f:
        json.dump(static_bboxes, f)
    print(f"  Static dedup: {before} → {after} ({before - after} removed)")


def post_process(run_dir):
    layout = _load_sensor_layout(run_dir)
    steps = [
        ("Semantic", lambda: convert_semantic(run_dir)),
        ("Depth", lambda: convert_depth(run_dir)),
        ("Original", lambda: convert_orin(run_dir, 95)),
        ("OCC", lambda: generate_occ(run_dir, layout)),
        ("Static Dedup", lambda: dedup_static_bboxes(run_dir)),
        ("Async Sim", lambda: simulate_async(run_dir)),
        ("Align Frames", lambda: align_frames(run_dir)),
        ("Remap IDs", lambda: remap_static_ids(run_dir)),
        ("Overall Filter", lambda: overall_filter_annotations(run_dir)),
        ("LiDAR Annotate", lambda: annotate_lidar(run_dir)),
        ("Camera Annotate", lambda: annotate_camera(run_dir)),
    ]
    for name, fn in tqdm(steps, desc="Post-processing", leave=True):
        fn()
    print("Post-processing complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/post_process.py <run_dir>")
        sys.exit(1)
    post_process(sys.argv[1])
