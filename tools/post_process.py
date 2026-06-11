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
                float(row["x"]), -float(row["y"]), float(row["z"]),
                -float(row["roll"]), float(row["pitch"]), -float(row["yaw"]))
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
                float(row["x"]), -float(row["y"]), float(row["z"]),
                -float(row["roll"]), float(row["pitch"]), -float(row["yaw"]))

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

    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if not ann_files:
        return

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

        temporal_keep = set()
        for s in recent_ids:
            temporal_keep |= s

        filtered = []
        kept_this_frame = set()
        for a in anns:
            loc = a["location"]; bb = a.get("bbox_3d", {})
            rot = a.get("rotation", {})
            hx = max(bb.get("x", 2.0) / 2, 1.0)
            hy = max(bb.get("y", 2.0) / 2, 0.5)
            hz = max(bb.get("z", 2.0) / 2, 0.5)
            dx = pts[:, 0] - loc["x"]
            dy = pts[:, 1] - loc["y"]
            dz = pts[:, 2] + 1.8 - loc["z"]
            yaw = m.radians(rot.get("yaw", 0))
            cr, sr = m.cos(yaw), m.sin(yaw)
            lx = dx * cr + dy * sr
            ly = -dx * sr + dy * cr
            pt_count = int(((np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(dz) <= hz)).sum())
            aid = a.get("actor_id", 0)
            if pt_count < min_pts and aid not in temporal_keep:
                occ_filtered_count += 1
                continue
            tid = str(a.get("type_id", ""))
            if tid.startswith("vehicle."):
                a["category"] = "vehicle"
            elif tid.startswith("walker.pedestrian."):
                a["category"] = "pedestrian"
            else:
                a["category"] = "vehicle"
            filtered.append(a)
            kept_this_frame.add(aid)

        out_path = os.path.join(out_dir, f"{frame:08d}.json")
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

def post_process(run_dir):
    layout = _load_sensor_layout(run_dir)
    steps = [
        ("Semantic", lambda: convert_semantic(run_dir)),
        ("Depth", lambda: convert_depth(run_dir)),
        ("Original", lambda: convert_orin(run_dir, 95)),
        ("OCC", lambda: generate_occ(run_dir, layout)),
        ("Align Frames", lambda: align_frames(run_dir)),
        ("Remap IDs", lambda: remap_static_ids(run_dir)),
        ("Overall Filter", lambda: overall_filter_annotations(run_dir)),
    ]
    for name, fn in tqdm(steps, desc="Post-processing", leave=True):
        fn()
    print("Post-processing complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/post_process.py <run_dir>")
        sys.exit(1)
    post_process(sys.argv[1])
