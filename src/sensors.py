import os
import csv
import json
import math
import numpy as np
from PIL import Image


def _sensor_output_dir(output_dir, spec):
    subdir = spec.get("subdir")
    if subdir:
        path = os.path.join(output_dir, spec["channel"], subdir)
    else:
        path = os.path.join(output_dir, spec["channel"])
    os.makedirs(path, exist_ok=True)
    return path


def attach_sensors(world, vehicle, bp_lib, layout, output_dir, gt_config=None):
    sensors = {}
    camera_specs = []
    occ_specs = []

    for spec in layout.get("sensors", []):
        if not spec.get("enabled", True):
            continue

        modality = spec["modality"]
        transform = _make_transform(spec.get("transform", {}))
        channel = spec["channel"]

        save_dir = _sensor_output_dir(output_dir, spec)

        if modality == "camera_rgb":
            sensors[spec["id"]] = _attach_camera(
                world, vehicle, bp_lib, spec, transform, save_dir
            )
            camera_specs.append(spec)

        elif modality == "camera_depth":
            sensors[spec["id"]] = _attach_depth_camera(
                world, vehicle, bp_lib, spec, transform, save_dir
            )

        elif modality == "camera_semantic":
            sensors[spec["id"]] = _attach_semantic_camera(
                world, vehicle, bp_lib, spec, transform, save_dir
            )

        elif modality == "lidar":
            sensors[spec["id"]] = _attach_lidar(
                world, vehicle, bp_lib, spec, transform, save_dir
            )

        elif modality == "lidar_semantic":
            sensors[spec["id"]] = _attach_semantic_lidar(
                world, vehicle, bp_lib, spec, transform, save_dir
            )

        elif modality == "gnss":
            sensors[spec["id"]] = _attach_gnss(
                world, vehicle, bp_lib, spec, transform, save_dir
            )

        elif modality == "imu":
            sensors[spec["id"]] = _attach_imu(
                world, vehicle, bp_lib, spec, transform, save_dir
            )

        elif modality == "occupancy":
            sensors[spec["id"]] = None  # pseudo-sensor, no CARLA actor
            occ_specs.append(spec)

    return sensors, camera_specs, occ_specs


def _compute_intrinsic(spec):
    out = spec.get("output", {})
    w = out.get("width", 1280)
    h = out.get("height", 720)
    hfov = math.radians(out.get("fov", 90))
    vfov = 2 * math.atan(math.tan(hfov / 2) * h / w)
    fx = w / (2 * math.tan(hfov / 2))
    fy = h / (2 * math.tan(vfov / 2))
    return {"fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2, "width": w, "height": h}


def _make_transform(t):
    return carla.Transform(
        carla.Location(x=t.get("x", 0), y=t.get("y", 0), z=t.get("z", 0)),
        carla.Rotation(
            roll=t.get("roll", 0), pitch=t.get("pitch", 0), yaw=t.get("yaw", 0)
        ),
    )


# ========== RGB Camera ==========

def _attach_camera(world, vehicle, bp_lib, spec, transform, save_dir):
    out = spec.get("output", {})
    bp = bp_lib.find(spec["blueprint"])
    bp.set_attribute("image_size_x", str(out.get("width", 1280)))
    bp.set_attribute("image_size_y", str(out.get("height", 720)))
    bp.set_attribute("fov", str(out.get("fov", 90)))
    rate = spec.get("rate_hz", 20)
    bp.set_attribute("sensor_tick", str(1.0 / rate))

    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.listen(_camera_callback(save_dir, spec["channel"]))
    return sensor


# Shared state: last frame captured per sensor channel
_sensor_frames = {}
_sensor_data = {}  # channel → latest point cloud array
_current_world_frame = 0


def set_world_frame(frame):
    global _current_world_frame
    _current_world_frame = frame


def get_sensor_frames():
    return _sensor_frames


def get_sensor_data():
    return _sensor_data


def _camera_callback(save_dir, channel):
    def cb(image):
        _sensor_frames[channel] = image.frame
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        np.save(os.path.join(save_dir, f"{_current_world_frame:08d}.npy"), array)

    return cb


# ========== Depth Camera ==========

def _attach_depth_camera(world, vehicle, bp_lib, spec, transform, save_dir):
    out = spec.get("output", {})
    bp = bp_lib.find(spec["blueprint"])
    bp.set_attribute("image_size_x", str(out.get("width", 1280)))
    bp.set_attribute("image_size_y", str(out.get("height", 720)))
    bp.set_attribute("fov", str(out.get("fov", 90)))
    rate = spec.get("rate_hz", 20)
    bp.set_attribute("sensor_tick", str(1.0 / rate))

    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.listen(_depth_callback(save_dir))
    return sensor


def _depth_callback(save_dir):

    def cb(image):
        # depth camera outputs BGRA uint8, 3 RGB channels encode logarithmic depth
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        h, w = image.height, image.width
        if array.size == h * w * 4:
            array = array.reshape((h, w, 4))
            # BGRA: B=idx0, G=idx1, R=idx2. CARLA encodes depth as R + G*256 + B*65536
            R = array[:, :, 2].astype(np.float64)
            G = array[:, :, 1].astype(np.float64)
            B = array[:, :, 0].astype(np.float64)
            depth = R + G * 256.0 + B * 65536.0
            depth = depth / 16777215.0 * 1000.0  # 256^3 - 1 = 16777215
        else:
            array = array.reshape((h, w))
            depth = array.astype(np.float32)
        np.save(os.path.join(save_dir, f"{_current_world_frame:08d}.npy"), depth)

    return cb


# ========== Semantic Segmentation Camera ==========

def _attach_semantic_camera(world, vehicle, bp_lib, spec, transform, save_dir):
    out = spec.get("output", {})
    bp = bp_lib.find(spec["blueprint"])
    bp.set_attribute("image_size_x", str(out.get("width", 1280)))
    bp.set_attribute("image_size_y", str(out.get("height", 720)))
    bp.set_attribute("fov", str(out.get("fov", 90)))
    rate = spec.get("rate_hz", 20)
    bp.set_attribute("sensor_tick", str(1.0 / rate))

    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.listen(_semantic_callback(save_dir))
    return sensor


# CARLA semantic tag → label mapping (CityScapes-compatible)
CARLA_SEMANTIC_LABELS = {
    0:  ("unlabeled",          (0, 0, 0)),
    1:  ("road",               (128, 64, 128)),
    2:  ("sidewalk",           (244, 35, 232)),
    3:  ("building",           (70, 70, 70)),
    4:  ("wall",               (102, 102, 156)),
    5:  ("fence",              (190, 153, 153)),
    6:  ("pole",               (153, 153, 153)),
    7:  ("traffic_light",      (250, 170, 30)),
    8:  ("traffic_sign",       (220, 220, 0)),
    9:  ("vegetation",         (107, 142, 35)),
    10: ("terrain",            (152, 251, 152)),
    11: ("sky",                (70, 130, 180)),
    12: ("pedestrian",         (220, 20, 60)),
    13: ("rider",              (255, 0, 0)),
    14: ("car",                (0, 0, 142)),
    15: ("truck",              (0, 0, 70)),
    16: ("bus",                (0, 60, 100)),
    17: ("train",              (0, 80, 100)),
    18: ("motorcycle",         (0, 0, 230)),
    19: ("bicycle",            (119, 11, 32)),
    20: ("static",             (110, 190, 160)),
    21: ("dynamic",            (170, 120, 50)),
    22: ("other",              (55, 90, 80)),
    23: ("water",              (45, 60, 150)),
    24: ("road_line",          (157, 234, 50)),
    25: ("ground",             (81, 0, 81)),
    26: ("bridge",             (150, 100, 100)),
    27: ("rail_track",         (230, 150, 140)),
    28: ("guard_rail",         (180, 165, 180)),
}


def _semantic_callback(save_dir):
    def cb(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        h, w = image.height, image.width
        if array.size == h * w * 4:
            array = array.reshape((h, w, 4))
            tags = array[:, :, 2].copy()
        else:
            tags = array.reshape((h, w))
        img = Image.fromarray(tags, mode="L")
        img.save(os.path.join(save_dir, f"{_current_world_frame:08d}.png"))

    return cb


# ========== LiDAR ==========

def _attach_lidar(world, vehicle, bp_lib, spec, transform, save_dir):
    out = spec.get("output", {})
    bp = bp_lib.find(spec["blueprint"])
    bp.set_attribute("channels", str(out.get("channels", 64)))
    bp.set_attribute("range", str(out.get("range", 100)))
    bp.set_attribute(
        "points_per_second", str(out.get("points_per_second", 56000))
    )
    bp.set_attribute(
        "rotation_frequency", str(out.get("rotation_frequency", 10))
    )
    rate = spec.get("rate_hz", 20)
    bp.set_attribute("sensor_tick", str(1.0 / rate))

    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.listen(_lidar_callback(save_dir, spec["channel"]))
    return sensor


def _lidar_callback(save_dir, channel):
    def cb(data):
        _sensor_frames[channel] = data.frame
        points = np.frombuffer(data.raw_data, dtype=np.float32).copy()
        points = points.reshape((-1, 4))
        points[:, 1] = -points[:, 1]  # Y: right → left (standard)
        _sensor_data[channel] = points
        np.save(os.path.join(save_dir, f"{_current_world_frame:08d}.npy"), points)

    return cb

def _attach_semantic_lidar(world, vehicle, bp_lib, spec, transform, save_dir):
    out = spec.get("output", {})
    bp = bp_lib.find(spec["blueprint"])
    bp.set_attribute("channels", str(out.get("channels", 64)))
    bp.set_attribute("range", str(out.get("range", 100)))
    bp.set_attribute(
        "points_per_second", str(out.get("points_per_second", 56000))
    )
    bp.set_attribute(
        "rotation_frequency", str(out.get("rotation_frequency", 10))
    )
    rate = spec.get("rate_hz", 20)
    bp.set_attribute("sensor_tick", str(1.0 / rate))

    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.listen(_semantic_lidar_callback(save_dir, spec["channel"], sensor))
    return sensor


def _semantic_lidar_callback(save_dir, channel, sensor):
    def cb(data):
        _sensor_frames[channel] = data.frame
        points = np.frombuffer(data.raw_data, dtype=np.float32).copy()
        points = points.reshape((-1, 6))  # x, y, z, cos_angle, obj_tag, obj_idx
        # Convert from CARLA world to sensor-local frame (X=fwd, Y=left, Z=up)
        t = sensor.get_transform()
        sx, sy, sz = t.location.x, t.location.y, t.location.z
        yaw = math.radians(t.rotation.yaw)
        cy, syaw = math.cos(yaw), math.sin(yaw)
        dx = points[:, 0] - sx
        dy = points[:, 1] - sy
        points[:, 0] = dx * cy + dy * syaw
        points[:, 1] = -(dx * syaw - dy * cy)
        points[:, 2] = points[:, 2] - sz
        _sensor_data[channel] = points
        np.save(os.path.join(save_dir, f"{_current_world_frame:08d}.npy"), points)

    return cb


# ========== GNSS ==========

def _attach_gnss(world, vehicle, bp_lib, spec, transform, save_dir):
    bp = bp_lib.find(spec["blueprint"])
    rate = spec.get("rate_hz", 20)
    bp.set_attribute("sensor_tick", str(1.0 / rate))
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.listen(_gnss_callback(save_dir))
    return sensor


def _gnss_callback(save_dir):
    csv_path = os.path.join(save_dir, "data.csv")
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["frame", "timestamp", "latitude", "longitude", "altitude"])

    def cb(data):
        writer.writerow(
            [_current_world_frame, data.timestamp, data.latitude, data.longitude, data.altitude]
        )
        f.flush()

    return cb


# ========== IMU ==========

def _attach_imu(world, vehicle, bp_lib, spec, transform, save_dir):
    bp = bp_lib.find(spec["blueprint"])
    rate = spec.get("rate_hz", 20)
    bp.set_attribute("sensor_tick", str(1.0 / rate))
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.listen(_imu_callback(save_dir))
    return sensor


def _imu_callback(save_dir):
    csv_path = os.path.join(save_dir, "data.csv")
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        [
            "frame", "timestamp",
            "accelerometer_x", "accelerometer_y", "accelerometer_z",
            "gyroscope_x", "gyroscope_y", "gyroscope_z",
            "compass",
        ]
    )

    def cb(data):
        writer.writerow(
            [
                _current_world_frame, data.timestamp,
                data.accelerometer.x, -data.accelerometer.y, data.accelerometer.z,
                -data.gyroscope.x, data.gyroscope.y, -data.gyroscope.z,
                data.compass,
            ]
        )
        f.flush()

    return cb


try:
    import carla
except ImportError:
    pass
