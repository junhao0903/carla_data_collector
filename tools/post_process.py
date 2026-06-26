#!/usr/bin/env python3
"""Post-processing: convert raw sensor data, generate OCC, filter annotations.

Usage: python tools/post_process.py <run_dir>
"""
import sys, os, json, math as m, glob, csv as _csv
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation

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
# Depth: .npy (float meters) → .png (uint16 cm)
# ══════════════════════════════════════════════════════════════════════

def convert_depth(run_dir):
    for cam_dir in sorted(glob.glob(os.path.join(run_dir, "CAM_*"))):
        src_dir = os.path.join(cam_dir, "depth")
        if not os.path.isdir(src_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
        if not npy_files:
            continue
        channel = os.path.basename(cam_dir)
        print(f"{channel}/depth: converting {len(npy_files)} .npy → .png (uint16 cm)")
        for npy_path in tqdm(npy_files, desc=f"{channel} depth", leave=True):
            arr = np.load(npy_path)
            if arr.ndim == 3 and arr.shape[2] == 4:
                # BGRA encoded depth → decode to meters
                R = arr[:, :, 2].astype(np.float32)
                G = arr[:, :, 1].astype(np.float32)
                B = arr[:, :, 0].astype(np.float32)
                depth = R + G * 256.0 + B * 65536.0
                depth = depth / 16777215.0 * 1000.0
            else:
                depth = arr  # already decoded (meters)
            # quantize: 1 unit = 1 cm, clamp 0-655.35m
            depth_u16 = np.clip(depth * 100.0, 0, 65535).astype(np.uint16)
            png_path = npy_path.replace(".npy", ".png")
            Image.fromarray(depth_u16).save(png_path)
            os.remove(npy_path)


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
            carla.Location(x=ex, y=-ey, z=ez),
            carla.Rotation(roll=eroll, pitch=-epitch, yaw=-eyaw))
        dynamic_list = []
        for a in anns:
            loc = a["location"];
            rot = a["rotation"];
            sz = a["bbox_3d"]
            # AD coords (Y=left) → CARLA world (Y=right)
            actor_tf = carla.Transform(
                carla.Location(x=loc["x"], y=-loc["y"], z=loc["z"]),
                carla.Rotation(roll=rot["roll"], pitch=-rot["pitch"], yaw=-rot["yaw"]))
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
        frames = {int(f.replace('.json', '').replace('.npy', '').replace('.jpg', '').replace('.png', ''))
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
                    os.remove(fp);
                    trimmed += 1
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


def _to_sensor_local(
        ax_ad, ay_ad, az_ad,
        aroll_ad, apitch_ad, ayaw_ad,
        ego,
        sensor_offsets
):
    """
    World(AD) -> Sensor

    return:
        sx, sy, sz,
        sroll, spitch, syaw
    """

    ex, ey, ez, eroll, epitch, eyaw = ego

    sx_off, sy_off, sz_off, \
        sroll_off, spitch_off, syaw_off = sensor_offsets

    # --------------------------------------------------
    # world -> ego
    # --------------------------------------------------
    R_world_ego = Rotation.from_euler(
        'xyz',
        [eroll, epitch, eyaw],
        degrees=True
    ).as_matrix()

    t_world_ego = np.array([ex, ey, ez])

    # --------------------------------------------------
    # ego -> sensor
    # --------------------------------------------------
    R_ego_sensor = Rotation.from_euler(
        'xyz',
        [
            np.degrees(sroll_off),
            np.degrees(spitch_off),
            np.degrees(syaw_off)
        ],
        degrees=True
    ).as_matrix()

    t_ego_sensor = np.array([
        sx_off,
        sy_off,
        sz_off
    ])

    # --------------------------------------------------
    # world -> sensor
    # --------------------------------------------------
    R_world_sensor = R_world_ego @ R_ego_sensor

    t_world_sensor = (
            t_world_ego +
            R_world_ego @ t_ego_sensor
    )

    # --------------------------------------------------
    # object position
    # --------------------------------------------------
    p_world = np.array([
        ax_ad,
        ay_ad,
        az_ad
    ])

    p_sensor = (
            R_world_sensor.T
            @
            (p_world - t_world_sensor)
    )

    # --------------------------------------------------
    # object orientation
    # --------------------------------------------------
    R_world_obj = Rotation.from_euler(
        'xyz',
        [aroll_ad, apitch_ad, ayaw_ad],
        degrees=True
    ).as_matrix()

    R_sensor_obj = (
            R_world_sensor.T
            @
            R_world_obj
    )

    sroll, spitch, syaw = Rotation.from_matrix(
        R_sensor_obj
    ).as_euler(
        'xyz',
        degrees=True
    )

    return (
        float(p_sensor[0]),
        float(p_sensor[1]),
        float(p_sensor[2]),
        float(sroll),
        float(spitch),
        float(syaw)
    )


def _count_points(pts, sx, sy, ayaw_s, hx, hy):
    """Count LiDAR points inside 2D bounding box (BEV projection)."""
    dx = pts[:, 0] - sx;
    dy = pts[:, 1] - sy
    yaw = m.radians(ayaw_s);
    cr, sr = m.cos(yaw), m.sin(yaw)
    lx = dx * cr + dy * sr;
    ly = -dx * sr + dy * cr
    return int(((np.abs(lx) <= hx) & (np.abs(ly) <= hy)).sum())


def overall_filter_annotations(run_dir):
    """Range-based filter in baselink coords."""
    filter_cfg = _load_filter_config()
    max_range = filter_cfg.get("max_range", filter_cfg.get("range", 100.0))

    ann_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if not os.path.isdir(ann_dir):
        print("Range filter: no ANNO/dynamic_actors, skipping")
        return
    valid_dir = os.path.join(run_dir, "ANNO", "valid")
    os.makedirs(valid_dir, exist_ok=True)

    # Load ego poses in AD coords
    ego_ad = {}
    ego_csv = os.path.join(run_dir, "TRAJ", "ego_trajectory.csv")
    if os.path.exists(ego_csv):
        with open(ego_csv) as f:
            for row in _csv.DictReader(f):
                ego_ad[int(row["frame"])] = (
                    float(row["x"]), float(row["y_left"]), float(row["z"]),
                    float(row["roll_left"]), float(row["pitch"]), float(row["yaw_left"]))

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

    total_dynamic_kept = 0
    total_dynamic_filtered = 0
    print(f"Range filter: {len(ann_files)} frames (max_range={max_range}m)")
    for ann_path in tqdm(ann_files, desc="Range filter", leave=True):
        frame = int(os.path.basename(ann_path).replace(".json", ""))
        ego = ego_ad.get(frame)
        if ego is None:
            continue
        ex, ey, ez = ego[0], ego[1], ego[2]

        with open(ann_path) as f:
            anns = json.load(f)

        filtered = []

        # dynamic actors
        for a in anns:
            ax = a["location"]["x"]
            ay = a["location"]["y"]
            az = a["location"]["z"]
            d = m.sqrt((ax - ex) ** 2 + (ay - ey) ** 2 + (az - ez) ** 2)
            if d > max_range:
                total_dynamic_filtered += 1
                continue
            total_dynamic_kept += 1
            filtered.append(a)

        # static bboxes within range
        for sb in static_bboxes:
            sx, sy, sz = sb["location"]["x"], sb["location"]["y"], sb["location"]["z"]
            d = m.sqrt((sx - ex) ** 2 + (sy - ey) ** 2 + (sz - ez) ** 2)
            if d <= max_range:
                entry = dict(sb)
                entry.setdefault("category", _category(entry.get("type_id", "")))
                filtered.append(entry)

        valid_path = os.path.join(valid_dir, f"{frame:08d}.json")
        with open(valid_path, "w") as f:
            json.dump(filtered, f)

    if total_dynamic_filtered > 0:
        print(f"  Range filter: {total_dynamic_filtered} filtered, {total_dynamic_kept} kept")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def _polygon_area(vertices):
    """Shoelace formula for convex polygon area."""
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0];
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _clip_polygon_to_rect(poly, xmin, xmax, ymin, ymax):
    """Sutherland-Hodgman clip convex polygon to axis-aligned rectangle."""
    edges = [
        (xmin, 0, lambda p: p >= xmin),
        (xmax, 0, lambda p: p <= xmax),
        (ymin, 1, lambda p: p >= ymin),
        (ymax, 1, lambda p: p <= ymax),
    ]
    for val, dim, inside_fn in edges:
        clipped = []
        for i in range(len(poly)):
            curr = poly[i];
            prev = poly[i - 1]
            c_in = inside_fn(curr[dim])
            p_in = inside_fn(prev[dim])
            if c_in:
                if not p_in:
                    t = (val - prev[dim]) / (curr[dim] - prev[dim])
                    clipped.append(prev + t * (curr - prev))
                clipped.append(curr)
            elif p_in:
                t = (val - prev[dim]) / (curr[dim] - prev[dim])
                clipped.append(prev + t * (curr - prev))
        if not clipped:
            return np.empty((0, 2))
        poly = np.array(clipped)
    return poly


def _load_cls_config():
    for search in ["config/cls/default.yaml"]:
        if os.path.exists(search):
            import yaml
            with open(search) as f:
                return yaml.safe_load(f) or {}
    return {}


def _cls_category(type_id):
    """Map CARLA type_id to cls config category key."""
    s = str(type_id)
    if "pedestrian" in s:
        return "pedestrian"
    for prefix in ["static."]:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("vehicle."):
        s = s[len("vehicle."):]
    s_lower = s.lower().replace("-", ".")
    if any(k in s_lower for k in ["motorcycle", "bicycle", "truck", "bus", "train"]):
        for k in ["motorcycle", "bicycle", "truck", "bus", "train"]:
            if k in s_lower:
                return k
    _motorcycle = {"harley", "kawasaki", "ninja", "vespa", "yamaha", "yzf"}
    _bicycle = {"crossbike", "diamondback", "gazelle", "omafiets"}
    _bus = {"fusorosa", "mitsubishi"}
    _truck = {"carlacola", "cybertruck", "ambulance", "firetruck", "european_hgv"}
    tokens = set(s_lower.split("."))
    if tokens & _bicycle: return "bicycle"
    if tokens & _motorcycle: return "motorcycle"
    if tokens & _bus: return "bus"
    if tokens & _truck: return "truck"
    return "vehicle"


def _ensure_bbox_3d(annotations):
    """Patch bbox_3d with cls config defaults for any dimension <= 0."""
    cls = _load_cls_config()
    for a in annotations:
        bb = a.get("bbox_3d", {})
        cat = _cls_category(a.get("type_id", ""))
        dims = cls.get(cat, {"x": 2.0, "y": 2.0, "z": 2.0})
        if bb.get("x", 0) <= 0:
            bb["x"] = dims["x"]
        if bb.get("y", 0) <= 0:
            bb["y"] = dims["y"]
        if bb.get("z", 0) <= 0:
            bb["z"] = dims["z"]
        a["bbox_3d"] = bb


def ensure_bbox_dims(run_dir):
    """Patch ANNO files: fill bbox_3d with cls defaults for dimensions <= 0."""
    import glob as _glob

    patch_count = 0
    # dynamic actors
    dyn_dir = os.path.join(run_dir, "ANNO", "dynamic_actors")
    if os.path.isdir(dyn_dir):
        for path in _glob.glob(os.path.join(dyn_dir, "*.json")):
            with open(path) as f:
                anns = json.load(f)
            _ensure_bbox_3d(anns)
            with open(path, "w") as f:
                json.dump(anns, f)
            patch_count += len(anns)

    # static bboxes
    static_path = os.path.join(run_dir, "ANNO", "static_bboxes.json")
    if os.path.exists(static_path):
        with open(static_path) as f:
            anns = json.load(f)
        _ensure_bbox_3d(anns)
        with open(static_path, "w") as f:
            json.dump(anns, f)
        patch_count += len(anns)

    if patch_count > 0:
        print(f"  Bbox patched: {patch_count} annotations")


def annotate_camera(run_dir):
    layout = _load_sensor_layout(run_dir)
    if not layout:
        return
    valid_dir = os.path.join(run_dir, "ANNO", "valid")
    if not os.path.isdir(valid_dir):
        return

    # ── CARLA semantic tags ──
    VEHICLE_TAGS = {14, 15, 16, 17, 18, 19, 13}  # car, truck, bus, train, motorcycle, bicycle, rider
    PEDESTRIAN_TAGS = {12}  # pedestrian

    def _target_tags(category):
        return PEDESTRIAN_TAGS if category == "pedestrian" else VEHICLE_TAGS

    def _vis_threshold(category):
        return 0.15 if category == "pedestrian" else 0.3

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
        dx = bb["x"] / 2
        dy = bb["y"] / 2
        dz = bb["z"] / 2

        return np.array([
            [dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, dz],
            [dx, -dy, -dz],
            [-dx, dy, dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
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
        if valid.sum() < 2:
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

        semantic_dir = os.path.join(run_dir, channel, "semantic")
        ann_dir = os.path.join(run_dir, channel, "annotations")
        os.makedirs(ann_dir, exist_ok=True)

        total_filtered = 0

        for fname in sorted(os.listdir(valid_dir)):
            if not fname.endswith(".json"):
                continue

            frame = int(fname.replace(".json", ""))
            if frame not in ego_ad:
                continue

            ego = ego_ad[frame]

            # ── load semantic image ──
            semantic_img = None
            sem_path = os.path.join(semantic_dir, f"{frame:08d}.png")
            if os.path.exists(sem_path):
                semantic_img = np.array(Image.open(sem_path))
            elif os.path.exists(sem_path.replace(".png", ".npy")):
                raw = np.load(sem_path.replace(".png", ".npy"))
                if raw.ndim == 3 and raw.shape[2] == 4:
                    semantic_img = raw[:, :, 2]
                else:
                    semantic_img = raw

            with open(os.path.join(valid_dir, fname)) as f:
                anns = json.load(f)

            projected = []

            for a in anns:

                # ── 1. to sensor frame ──
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

                if sx <= 0.1:
                    continue

                # ── 2. project 3D bbox corners ──
                corners = make_corners(a.get("bbox_3d", {}))

                R = Rotation.from_euler(
                    'xyz',
                    [sroll, spitch, syaw],
                    degrees=True
                ).as_matrix()

                corners = (R @ corners.T).T + np.array([sx, sy, sz])

                proj = project(K, corners)
                if proj is None:
                    continue

                u, v, valid_mask = proj
                u = u[valid_mask]
                v = v[valid_mask]

                # ── 3. original bbox (before clipping) ──
                xmin_o = u.min()
                xmax_o = u.max()
                ymin_o = v.min()
                ymax_o = v.max()

                original_area = (xmax_o - xmin_o) * (ymax_o - ymin_o)
                if original_area <= 0:
                    continue

                # ── 4. clip bbox to image bounds ──
                xmin_c = max(0, int(xmin_o))
                ymin_c = max(0, int(ymin_o))
                xmax_c = min(w - 1, int(xmax_o))
                ymax_c = min(h - 1, int(ymax_o))

                if xmin_c >= xmax_c or ymin_c >= ymax_c:
                    continue

                clip_area = (xmax_c - xmin_c) * (ymax_c - ymin_c)

                # ── 5. truncation ──
                truncation = 1.0 - clip_area / original_area

                # ── 6. semantic visibility ──
                visibility = 1.0
                if semantic_img is not None:
                    roi = semantic_img[ymin_c:ymax_c, xmin_c:xmax_c]
                    tags = _target_tags(a.get("category", "vehicle"))
                    visible_pixels = sum(int((roi == t).sum()) for t in tags)
                    visibility = visible_pixels / clip_area

                # ── 7. filter by visibility ──
                if visibility < _vis_threshold(a.get("category", "vehicle")):
                    total_filtered += 1
                    continue

                # ── 8. save ──
                entry = dict(a)
                entry["bbox_2d"] = [xmin_c, ymin_c, xmax_c, ymax_c]
                entry["bbox_original"] = [int(xmin_o), int(ymin_o), int(xmax_o), int(ymax_o)]
                entry["truncation"] = round(truncation, 4)
                entry["visibility"] = round(visibility, 4)
                entry["location"] = {
                    "x": float(sx),
                    "y": float(sy),
                    "z": float(sz),
                }

                projected.append(entry)

            out_path = os.path.join(ann_dir, f"{frame:08d}.json")
            with open(out_path, "w") as f:
                json.dump(projected, f)

        print(f"[OK] {channel} (semantic filtered: {total_filtered})")


def annotate_lidar(run_dir):
    """Project ANNO/valid annotations into each LiDAR sensor frame with FOV + point-count filter."""
    layout = _load_sensor_layout(run_dir)
    if not layout:
        return
    valid_dir = os.path.join(run_dir, "ANNO", "valid")
    if not os.path.isdir(valid_dir):
        return

    filter_cfg = _load_filter_config()
    min_pts_floor = filter_cfg.get("min_pts", 3)
    pts_per_m3 = filter_cfg.get("pts_per_m3", 5)

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

        pts_dir = os.path.join(run_dir, channel, "original")
        ann_dir = os.path.join(run_dir, channel, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        count = 0
        total_pts_filtered = 0

        for fname in sorted(os.listdir(valid_dir)):
            frame = int(fname.replace(".json", ""))
            ego = ego_ad.get(frame)
            if ego is None:
                continue

            # Load LiDAR point cloud
            pts = None
            pts_path = os.path.join(pts_dir, f"{frame:08d}.npy")
            if os.path.exists(pts_path):
                pts = np.load(pts_path)

            with open(os.path.join(valid_dir, fname)) as f:
                anns = json.load(f)

            projected = []
            for a in anns:
                rot = a.get("rotation", {})
                sx, sy, sz, sroll, spitch, syaw = _to_sensor_local(
                    a["location"]["x"], a["location"]["y"], a["location"]["z"],
                    rot.get("roll", 0), rot.get("pitch", 0), rot.get("yaw", 0),
                    ego, lidar_offsets)

                # ── FOV filter ──
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

                # ── point-count filter ──
                if pts is not None:
                    bb = a.get("bbox_3d", {})
                    hx = bb["x"] / 2
                    hy = bb["y"] / 2
                    hz = bb["z"] / 2
                    n = _count_points(pts, sx, sy, syaw, hx, hy)
                    volume = (2 * hx) * (2 * hy) * (2 * hz)
                    threshold = max(min_pts_floor, int(volume * pts_per_m3))
                    if n < threshold:
                        total_pts_filtered += 1
                        continue

                entry = dict(a)
                entry["location"] = {"x": sx, "y": sy, "z": sz}
                entry["rotation"] = {"roll": sroll, "pitch": spitch, "yaw": syaw}
                projected.append(entry)

            out_path = os.path.join(ann_dir, f"{frame:08d}.json")
            with open(out_path, "w") as f:
                json.dump(projected, f)
            count += 1

        print(f"  {channel}/annotations: {count} frames (pts filtered: {total_pts_filtered})")


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
                     if not any(((sb["location"]["x"] - a["location"]["x"]) ** 2 +
                                 (sb["location"]["y"] - a["location"]["y"]) ** 2) ** 0.5 < 2.0
                                for a in dyn_actors)
                     and not (ego_pos is not None and
                              ((sb["location"]["x"] - ego_pos[0]) ** 2 +
                               (sb["location"]["y"] - ego_pos[1]) ** 2) ** 0.5 < 0.5)]
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
        ("Bbox Patch", lambda: ensure_bbox_dims(run_dir)),
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
