import json
import math
import os
import random
import time
import signal
from datetime import datetime

import numpy as np
from PIL import Image


class DataCollector:
    def __init__(self, config, carla_server):
        self._config = config
        self._server = carla_server
        self._world = None
        self._vehicle = None
        self._npc_vehicles = []
        self._sensors = {}
        self._rgb_sensors = {}  # channel → sensor actor
        self._lidar_sensors = {}  # channel → sensor actor
        self._camera_specs = []
        self._lidar_specs = []
        self._output_dir = None
        self._frame_annotations = {}
        self._occ_ann_dir = None

    def run(self):
        self._server.connect()
        carla = self._server.carla
        client = self._server.client

        town = self._config["carla"].get("map")
        if town:
            self._world = client.load_world(town)
        else:
            self._world = client.get_world()

        fps = self._config["carla"].get("fps", 20)
        sync = self._config.get("collection", {}).get("synchronous", True)

        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 1.0 / fps if sync else 0.0
        self._world.apply_settings(settings)

        self._ensure_static_occ(self._world)

        bp_lib = self._world.get_blueprint_library()
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(sync)
        tm.set_random_device_seed(0)

        weather = self._config["carla"].get("weather", "ClearNoon")
        if hasattr(carla.WeatherParameters, weather):
            self._world.set_weather(getattr(carla.WeatherParameters, weather))
        else:
            print(f"Unknown weather preset '{weather}', using ClearNoon")
            self._world.set_weather(carla.WeatherParameters.ClearNoon)

        self._vehicle = self._spawn_vehicle(self._world, bp_lib)
        self._vehicle.set_autopilot(True, tm.get_port())

        self._npc_vehicles = self._spawn_npc_vehicles(self._world, bp_lib, tm, self._vehicle)
        print(f"Spawned {len(self._npc_vehicles)} NPC vehicles")

        col = self._config.get("collection", {})
        start_loc = self._vehicle.get_transform().location
        warmup = col.get("warmup_ticks", 10)
        for _ in range(warmup):
            self._world.tick()

        self._wait_for_motion(self._world, self._vehicle, col, start_loc)
        print("Vehicle moving, starting collection")

        self._output_dir = self._make_output_dir()
        # Save sensor layout for post-processing (contains vis flags)
        import yaml as _yaml
        with open(os.path.join(self._output_dir, "sensor_layout.yaml"), "w") as _f:
            _yaml.dump(self._config.get("_sensor_layout", {}), _f)
        print(f"Saving data to {self._output_dir}")

        traj_dir = os.path.join(self._output_dir, "TRAJ")
        os.makedirs(traj_dir, exist_ok=True)
        self._ego_csv = os.path.join(traj_dir, "ego_trajectory.csv")
        self._ego_f = open(self._ego_csv, "w", newline="")
        import csv as csv_module
        self._ego_writer = csv_module.writer(self._ego_f)
        self._ego_writer.writerow(["frame", "x", "y_left", "z", "roll_left", "pitch", "yaw_left",
                                    "cam_x", "cam_y_left", "cam_z", "cam_roll_left", "cam_pitch", "cam_yaw_left"])

        from .sensors import attach_sensors
        sensor_layout = self._build_sensor_layout(fps)
        self._sensors, self._camera_specs, occ_specs = attach_sensors(
            self._world, self._vehicle, bp_lib, sensor_layout, self._output_dir
        )
        self._occ_spec = occ_specs[0] if occ_specs else None
        # Track LiDAR sensors for annotation
        for spec in sensor_layout.get("sensors", []):
            if spec.get("enabled", True) and spec["modality"] in ("lidar", "lidar_semantic"):
                sensor_id = spec["id"]
                if sensor_id in self._sensors:
                    self._lidar_sensors[spec["channel"]] = self._sensors[sensor_id]
                    self._lidar_specs.append(spec)

        # Build channel → sensor lookup for camera transform retrieval
        for spec in self._camera_specs:
            sensor_id = spec["id"]
            if sensor_id in self._sensors:
                self._rgb_sensors[spec["channel"]] = self._sensors[sensor_id]

        # Spawn dedicated filter LiDAR if configured (independent of sensor layout)
        self._filter_lidar = None
        self._filter_lidar_channel = "__filter__"
        self._filter_min_pts = 10
        filter_cfg = self._config.get("_filter_config", {})
        if filter_cfg.get("enabled", True):
            self._setup_filter_lidar(self._world, bp_lib, self._vehicle, filter_cfg)
            self._filter_min_pts = filter_cfg.get("min_points", 10)

        self._init_annotations()
        print(f"Attached sensors: {', '.join(self._sensors.keys())}")

        # Sync tick: ensure all sensors (especially derived depth/semantic)
        # fire at least once so frame counts align across modalities
        if sync:
            self._world.tick()

        duration = self._config.get("collection", {}).get("duration_seconds", 0)
        start_time = time.time()
        try:
            while True:
                if sync:
                    frame = self._world.tick()
                else:
                    time.sleep(1.0 / fps)
                    frame = self._world.get_snapshot().frame

                self._write_annotations_for_new_frames()
                self._record_ego_pose(frame)
                self._save_occ_gt_data(frame)

                if duration > 0 and (time.time() - start_time) >= duration:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted, shutting down...")
        finally:
            self._cleanup()

        self._convert_npy_to_jpg()

    def _save_occ_gt_data(self, frame):
        """Save per-frame GT annotations (ego frame, left-hand coords).

        Same coordinate system as LiDAR annotations:
          X=forward, Y=left, Z=up; roll=左翼下沉+, pitch=抬头+, yaw=机头左转+
        Origin: ego vehicle center.

        Includes dynamic actors + spatially-filtered static level_bbs.
        """
        if self._occ_ann_dir is None:
            return
        import json as _json

        occ_cfg = self._occ_spec.get("output", {}) if self._occ_spec else {}
        pc = occ_cfg.get("pc_range", [-50, -50, -5, 50, 50, 3])
        margin = occ_cfg.get("static_filter_margin", 10.0)

        ego_t = self._vehicle.get_transform()
        ex, ey, ez = ego_t.location.x, ego_t.location.y, ego_t.location.z
        eyaw = math.radians(ego_t.rotation.yaw)
        cy, sy = math.cos(eyaw), math.sin(eyaw)

        def _world_to_ego(wx, wy, wz, wroll, wpitch, wyaw):
            """CARLA world (X=fwd,Y=right) → ego frame (X=fwd,Y=left)."""
            dx, dy = wx - ex, wy - ey
            return {
                "x": dx * cy + dy * sy,
                "y": dx * sy - dy * cy,      # Y=left
                "z": wz - ez,
                "roll": -wroll,
                "pitch": wpitch,
                "yaw": -(wyaw - ego_t.rotation.yaw),
            }

        annotations = []
        ego_id = self._vehicle.id

        # Dynamic actors
        for actor in self._world.get_actors():
            try:
                if not actor.is_alive or actor.id == ego_id:
                    continue
                type_id = str(actor.type_id)
                if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
                    continue
                extent = actor.bounding_box.extent
                t = actor.get_transform()
                vel = actor.get_velocity()
                p = _world_to_ego(t.location.x, t.location.y, t.location.z,
                                  t.rotation.roll, t.rotation.pitch, t.rotation.yaw)
                v = _world_to_ego(vel.x, vel.y, vel.z, 0, 0, 0)
                annotations.append({
                    "actor_id": int(actor.id),
                    "type_id": type_id,
                    "location": {"x": p["x"], "y": p["y"], "z": p["z"]},
                    "rotation": {"roll": p["roll"], "pitch": p["pitch"], "yaw": p["yaw"]},
                    "extent": {"x": float(extent.x), "y": float(extent.y), "z": float(extent.z)},
                    "velocity": {"x": v["x"], "y": v["y"], "z": v["z"]},
                })
            except RuntimeError:
                continue

        # Static level_bbs (same categories as camera annotations)
        static_label_map = {
            self._server.carla.CityObjectLabel.Car: "static.car",
            self._server.carla.CityObjectLabel.Truck: "static.truck",
            self._server.carla.CityObjectLabel.Bus: "static.bus",
            self._server.carla.CityObjectLabel.Train: "static.train",
            self._server.carla.CityObjectLabel.Motorcycle: "static.motorcycle",
            self._server.carla.CityObjectLabel.Bicycle: "static.bicycle",
            self._server.carla.CityObjectLabel.Pedestrians: "static.pedestrian",
        }
        for obj_label, type_id in static_label_map.items():
            try:
                bbs = self._world.get_level_bbs(obj_label)
            except RuntimeError:
                continue
            for bb in bbs:
                # Spatial filter in ego frame (before conversion)
                dx, dy = bb.location.x - ex, bb.location.y - ey
                fwd = dx * cy + dy * sy
                lft = dx * sy - dy * cy
                if not (pc[0] - margin <= fwd <= pc[3] + margin and
                        pc[1] - margin <= lft <= pc[4] + margin):
                    continue
                p = _world_to_ego(bb.location.x, bb.location.y, bb.location.z,
                                  bb.rotation.roll, bb.rotation.pitch, bb.rotation.yaw)
                annotations.append({
                    "actor_id": -1,
                    "type_id": type_id,
                    "location": {"x": p["x"], "y": p["y"], "z": p["z"]},
                    "rotation": {"roll": p["roll"], "pitch": p["pitch"], "yaw": p["yaw"]},
                    "extent": {"x": float(bb.extent.x), "y": float(bb.extent.y), "z": float(bb.extent.z)},
                    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                })

        ann_path = os.path.join(self._occ_ann_dir, f"{frame:08d}.json")
        with open(ann_path, "w") as f:
            _json.dump(annotations, f)

    @staticmethod
    def _ensure_static_occ(world):
        """Generate static OCC for current map if not already cached."""
        town = world.get_map().name.split("/")[-1]
        npy_path = os.path.join("map", f"{town}_static_occ.npy")
        if os.path.exists(npy_path):
            return
        print(f"Building static OCC for {town} (one-time, ~1-2 min)...")
        os.makedirs("map", exist_ok=True)
        from .occ_generator import build_static_occ
        # Use a map-covering range from spawn points
        spawn_pts = world.get_map().get_spawn_points()
        xs = [sp.location.x for sp in spawn_pts]
        ys = [sp.location.y for sp in spawn_pts]
        margin = 100.0
        pc_range = [min(xs) - margin, min(ys) - margin, -5.0,
                     max(xs) + margin, max(ys) + margin, 3.0]
        voxel_size = [0.5, 0.5, 0.5]
        shape = [int(round((pc_range[3] - pc_range[0]) / voxel_size[0])),
                 int(round((pc_range[4] - pc_range[1]) / voxel_size[1])),
                 int(round((pc_range[5] - pc_range[2]) / voxel_size[2]))]
        occ = build_static_occ(world, pc_range, voxel_size, shape)
        np.save(npy_path, occ)
        import json as _json
        with open(npy_path.replace(".npy", ".json"), "w") as f:
            _json.dump({"map": town, "pc_range": pc_range,
                         "voxel_size": voxel_size, "shape": shape}, f)
        print(f"Static OCC saved to {npy_path} ({occ.nbytes / 1e6:.0f} MB)")

    def _setup_filter_lidar(self, world, bp_lib, vehicle, cfg):
        bp = bp_lib.find(cfg.get("blueprint", "sensor.lidar.ray_cast"))
        bp.set_attribute("channels", str(cfg.get("channels", 64)))
        bp.set_attribute("range", str(cfg.get("range", 100)))
        bp.set_attribute("points_per_second", str(cfg.get("points_per_second", 56000)))
        bp.set_attribute("rotation_frequency", str(cfg.get("rotation_frequency", 10)))
        rate = cfg.get("rate_hz", 20)
        bp.set_attribute("sensor_tick", str(1.0 / rate))

        from .sensors import _make_transform
        transform = _make_transform(cfg.get("transform", {}))
        sensor = world.spawn_actor(bp, transform, attach_to=vehicle)

        from .sensors import _sensor_data, _sensor_frames
        channel = self._filter_lidar_channel

        save_dir = None
        if cfg.get("output", False):
            save_dir = os.path.join(self._output_dir, "LIDAR_FILTER", "original")
            os.makedirs(save_dir, exist_ok=True)

        def _filter_lidar_cb(data):
            _sensor_frames[channel] = data.frame
            points = np.frombuffer(data.raw_data, dtype=np.float32).copy()
            points = points.reshape((-1, 4))
            points[:, 1] = -points[:, 1]
            _sensor_data[channel] = points
            if save_dir is not None:
                np.save(os.path.join(save_dir, f"{data.frame:08d}.npy"), points)

        sensor.listen(_filter_lidar_cb)
        self._filter_lidar = sensor
        self._sensors["__filter_lidar__"] = sensor
        if save_dir is not None:
            self._lidar_sensors["LIDAR_FILTER"] = sensor

    def _build_sensor_layout(self, world_fps):
        layout = self._config.get("_sensor_layout", {})
        sensors = list(layout.get("sensors", []))

        # Auto-tag sensors with subdir for grouped output (skip pseudo-sensors)
        for s in sensors:
            if s.get("modality") in ("camera_rgb", "lidar", "lidar_semantic"):
                s.setdefault("subdir", "original")

        for s in layout.get("sensors", []):
            if s.get("modality") != "camera_rgb" or not s.get("enabled", True):
                continue
            if s.get("depth", True):
                sensors.append({
                    "id": s["id"] + "_depth",
                    "channel": s["channel"],
                    "subdir": "depth",
                    "modality": "camera_depth",
                    "blueprint": "sensor.camera.depth",
                    "enabled": True,
                    "rate_hz": s.get("rate_hz", world_fps),
                    "transform": s.get("transform", {}),
                    "output": s.get("output", {}),
                })
            if s.get("semantic", True):
                sensors.append({
                    "id": s["id"] + "_semantic",
                    "channel": s["channel"],
                    "subdir": "semantic",
                    "modality": "camera_semantic",
                    "blueprint": "sensor.camera.semantic_segmentation",
                    "enabled": True,
                    "rate_hz": s.get("rate_hz", world_fps),
                    "transform": s.get("transform", {}),
                    "output": s.get("output", {}),
                })

        return {"sensors": sensors}

    def _init_annotations(self):
        self._frame_annotations = {}
        for spec in self._camera_specs:
            channel = spec["channel"]
            ann_dir = os.path.join(self._output_dir, channel, "annotations")
            os.makedirs(ann_dir, exist_ok=True)
            from .sensors import _compute_intrinsic
            self._frame_annotations[channel] = {
                "dir": ann_dir,
                "intrinsic": _compute_intrinsic(spec),
            }

        for spec in self._lidar_specs:
            channel = spec["channel"]
            ann_dir = os.path.join(self._output_dir, channel, "annotations")
            os.makedirs(ann_dir, exist_ok=True)

        if self._filter_lidar is not None:
            filter_cfg = self._config.get("_filter_config", {})
            if filter_cfg.get("output", False):
                ann_dir = os.path.join(self._output_dir, "LIDAR_FILTER", "annotations")
                os.makedirs(ann_dir, exist_ok=True)

        if self._occ_spec is not None:
            from .sensors import _sensor_output_dir
            occ_save_dir = _sensor_output_dir(self._output_dir, self._occ_spec)
            self._occ_ann_dir = os.path.join(self._output_dir, "OCC", "annotations")
            os.makedirs(self._occ_ann_dir, exist_ok=True)
            # Save OCC grid params + metadata for post-processing
            import json as _json
            occ_cfg = self._occ_spec.get("output", {})
            occ_cfg["static_occ_dir"] = occ_cfg.get("static_occ_dir", "map")
            occ_cfg["map"] = self._config["carla"].get("map", "")
            with open(os.path.join(self._output_dir, "OCC", "occ_metadata.json"), "w") as _f:
                _json.dump(occ_cfg, _f)

    def _write_annotations_for_new_frames(self):
        from .sensors import get_sensor_frames, get_sensor_data

        # Snapshot lidar data once to avoid race with callback threads
        # Prefer dedicated filter LiDAR, fall back to data-collection LiDARs
        lidar_points = None
        min_pts = self._filter_min_pts
        ego_tf = self._vehicle.get_transform()
        lidar_tf = None
        if min_pts > 0:
            pts = get_sensor_data().get(self._filter_lidar_channel)
            if pts is not None:
                lidar_points = pts
                if self._filter_lidar and self._filter_lidar.is_alive:
                    lidar_tf = self._filter_lidar.get_transform()
            else:
                for lidar_spec in self._lidar_specs:
                    ch = lidar_spec["channel"]
                    pts = get_sensor_data().get(ch)
                    if pts is not None:
                        lidar_points = pts
                        sensor = self._lidar_sensors.get(ch)
                        if sensor and sensor.is_alive:
                            lidar_tf = sensor.get_transform()
                        break

        for spec in self._camera_specs:
            channel = spec["channel"]
            frame = get_sensor_frames().get(channel)
            if frame is None:
                continue
            ann_path = os.path.join(
                self._frame_annotations[channel]["dir"], f"{frame:08d}.json"
            )
            if os.path.exists(ann_path):
                continue
            # Only use lidar filter if lidar frame is close to camera frame
            lidar_frame = get_sensor_frames().get(self._filter_lidar_channel)
            if lidar_frame is None and self._lidar_specs:
                lidar_frame = get_sensor_frames().get(self._lidar_specs[0]["channel"])
            if lidar_frame is not None and abs(lidar_frame - frame) <= 2:
                lp, mp = lidar_points, min_pts
            else:
                lp, mp = None, 0
            self._write_annotations(frame, spec, channel, ann_path, lp, mp, ego_tf, lidar_tf)

        for spec in self._lidar_specs:
            channel = spec["channel"]
            frame = get_sensor_frames().get(channel)
            if frame is None:
                continue
            ann_dir = os.path.join(self._output_dir, channel, "annotations")
            ann_path = os.path.join(ann_dir, f"{frame:08d}.json")
            if os.path.exists(ann_path):
                continue
            self._write_lidar_annotations(ann_path, channel, spec, frame, lidar_points, min_pts, ego_tf, lidar_tf)

        # Write filter LiDAR annotations if output enabled
        if self._filter_lidar is not None and self._config.get("_filter_config", {}).get("output", False):
            filter_frame = get_sensor_frames().get(self._filter_lidar_channel)
            if filter_frame is not None:
                ann_dir = os.path.join(self._output_dir, "LIDAR_FILTER", "annotations")
                ann_path = os.path.join(ann_dir, f"{filter_frame:08d}.json")
                if not os.path.exists(ann_path):
                    filter_cfg = self._config.get("_filter_config", {})
                    filter_spec = {
                        "channel": "LIDAR_FILTER",
                        "output": {
                            "range": filter_cfg.get("range", 100),
                            "upper_fov": filter_cfg.get("upper_fov", 10),
                            "lower_fov": filter_cfg.get("lower_fov", -30),
                            "horizontal_fov": filter_cfg.get("horizontal_fov", 360),
                        }
                    }
                    self._write_lidar_annotations(ann_path, "LIDAR_FILTER", filter_spec,
                                                  filter_frame, lidar_points, min_pts, ego_tf, lidar_tf)

    @staticmethod
    def _count_points_in_actor_bbox(points, actor, ego_tf, lidar_tf):
        """points in lidar local frame (Y flipped to left). Count XY points inside actor bbox."""
        # Actor bbox center in CARLA world
        bbox = actor.bounding_box
        t = actor.get_transform()
        ayaw = math.radians(t.rotation.yaw)
        acos, asin = math.cos(ayaw), math.sin(ayaw)
        bbl = bbox.location
        cx_w = t.location.x + bbl.x * acos - bbl.y * asin
        cy_w = t.location.y + bbl.x * asin + bbl.y * acos
        ex, ey = bbox.extent.x, bbox.extent.y

        # Convert actor center to ego vehicle local frame (forward/right/up)
        eyaw = math.radians(ego_tf.rotation.yaw)
        ecos, esin = math.cos(eyaw), math.sin(eyaw)
        dx_w = cx_w - ego_tf.location.x
        dy_w = cy_w - ego_tf.location.y
        cx_ego = dx_w * ecos + dy_w * esin
        cy_ego = -dx_w * esin + dy_w * ecos  # CARLA Y=right in ego frame

        # LiDAR mount transform in ego frame (assume rotation=0, only Z offset matters)
        lidar_z_offset = lidar_tf.location.z - ego_tf.location.z if lidar_tf else 0.0

        # LiDAR points in ego frame: points[:,0]=forward, points[:,1]=left (flipped), points[:,2]=up
        # Remove LiDAR Z offset to align with ego origin
        px_ego = points[:, 0]  # forward (same as ego X)
        py_ego = points[:, 1]  # left (flipped from CARLA right. Ego right = -left)
        pz_ego = points[:, 2] - lidar_z_offset

        # Actor center in ego frame with Y=left convention (matching flipped lidar Y)
        cy_ego_left = -cy_ego  # CARLA right → left

        # Vector from actor center to points in ego frame (Y=left)
        dx = px_ego - cx_ego
        dy = py_ego - cy_ego_left

        # Actor yaw in ego frame
        actor_yaw_ego = ayaw - eyaw
        acos2, asin2 = math.cos(actor_yaw_ego), math.sin(actor_yaw_ego)

        # Rotate to actor local frame
        lx = dx * acos2 + dy * asin2
        ly = -dx * asin2 + dy * acos2

        inside = (np.abs(lx) <= ex) & (np.abs(ly) <= ey)
        return int(inside.sum())

    def _write_annotations(self, frame, spec, channel, ann_path, lidar_points, min_pts, ego_tf, lidar_tf):
        actors = list(self._world.get_actors())
        annotations = []
        ego_id = self._vehicle.id

        for actor in actors:
            try:
                if not actor.is_alive or actor.id == ego_id:
                    continue
                type_id = str(actor.type_id)
                if type_id.startswith("vehicle."):
                    category = "vehicle"
                elif type_id.startswith("walker.pedestrian."):
                    category = "pedestrian"
                else:
                    continue

                if lidar_points is not None and self._count_points_in_actor_bbox(lidar_points, actor, ego_tf, lidar_tf) < min_pts:
                    continue

                bbox = actor.bounding_box.extent
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                annotations.append({
                    "actor_id": int(actor.id),
                    "category": category,
                    "type_id": type_id,
                    "bbox_3d": {
                        "x": float(bbox.x * 2), "y": float(bbox.y * 2), "z": float(bbox.z * 2),
                    },
                    "location": {
                        "x": float(transform.location.x),
                        "y": float(transform.location.y),
                        "z": float(transform.location.z),
                    },
                    "rotation": {
                        "roll": -float(transform.rotation.roll),
                        "pitch": float(transform.rotation.pitch),
                        "yaw": float(transform.rotation.yaw),
                    },
                    "velocity": {
                        "x": float(velocity.x),
                        "y": float(velocity.y),
                        "z": float(velocity.z),
                    },
                })
            except RuntimeError:
                continue

        # Add static level vehicles (skip if overlapping with dynamic actors)
        dynamic_locs = []
        for a in annotations:
            dynamic_locs.append((a["location"]["x"], a["location"]["y"]))

        label_map = {
            self._server.carla.CityObjectLabel.Car: "static_car",
            self._server.carla.CityObjectLabel.Truck: "static_truck",
            self._server.carla.CityObjectLabel.Bus: "static_bus",
            self._server.carla.CityObjectLabel.Train: "static_train",
            self._server.carla.CityObjectLabel.Motorcycle: "static_motorcycle",
            self._server.carla.CityObjectLabel.Bicycle: "static_bicycle",
            self._server.carla.CityObjectLabel.Pedestrians: "static_pedestrian",
        }
        for label, cat in label_map.items():
            try:
                bbs = self._world.get_level_bbs(label)
            except RuntimeError:
                continue
            for bb in bbs:
                # Skip if overlaps with any dynamic actor
                bx, by = bb.location.x, bb.location.y
                overlap = False
                for dx, dy in dynamic_locs:
                    if abs(bx - dx) < 3.0 and abs(by - dy) < 3.0:
                        overlap = True
                        break
                if overlap:
                    continue
                if lidar_points is not None:
                    hx, hy = bb.extent.x, bb.extent.y
                    eyaw = math.radians(ego_tf.rotation.yaw)
                    ecos, esin = math.cos(eyaw), math.sin(eyaw)
                    dx_w = bb.location.x - ego_tf.location.x
                    dy_w = bb.location.y - ego_tf.location.y
                    cx_ego = dx_w * ecos + dy_w * esin
                    cy_ego = -dx_w * esin + dy_w * ecos
                    cy_ego_left = -cy_ego
                    n_static = int(((np.abs(lidar_points[:, 0] - cx_ego) <= hx) & \
                                    (np.abs(lidar_points[:, 1] - cy_ego_left) <= hy)).sum())
                    if n_static < min_pts:
                        continue
                annotations.append({
                    "actor_id": -1,
                    "category": cat,
                    "type_id": str(label).split('.')[-1],
                    "bbox_3d": {
                        "x": float(bb.extent.x * 2),
                        "y": float(bb.extent.y * 2),
                        "z": float(bb.extent.z * 2),
                    },
                    "location": {"x": bb.location.x, "y": bb.location.y, "z": bb.location.z - bb.extent.z},
                    "rotation": {"roll": -float(bb.rotation.roll), "pitch": float(bb.rotation.pitch), "yaw": float(bb.rotation.yaw)},
                    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                })

        cam_transform = self._get_camera_world_transform(channel)
        K = self._frame_annotations[channel]["intrinsic"]

        bboxes_2d = []
        for ann in annotations:
            bbox_2d = self._project_bbox(ann, cam_transform, spec, K)
            bw = bbox_2d[2] - bbox_2d[0] if bbox_2d else 0
            bh = bbox_2d[3] - bbox_2d[1] if bbox_2d else 0
            img_area = K["width"] * K["height"]
            img_short = min(K["width"], K["height"])
            tiny = (bw < 16 and bh < 16)
            low_ratio = (bw * bh) / img_area < 0.0001
            thin = max(bw, bh) < img_short / 50
            # Theoretical projection size vs actual: discard truncated bboxes
            loc_w = ann["location"]
            ext_w = ann["bbox_3d"]
            dx_w = loc_w["x"] - cam_transform.location.x
            dy_w = loc_w["y"] - cam_transform.location.y
            dz_w = (loc_w["z"] + ext_w["z"] / 2) - cam_transform.location.z
            dist = math.sqrt(dx_w*dx_w + dy_w*dy_w + dz_w*dz_w)
            expected_w = K["fx"] * ext_w["y"] / max(dist, 0.1)   # width in image (Y axis → u)
            expected_h = K["fy"] * ext_w["z"] / max(dist, 0.1)   # height in image
            expected_area = expected_w * expected_h
            truncated = (bw * bh) < expected_area * 0.3 if expected_area > 0 else True
            if bbox_2d is not None and not (tiny or low_ratio or thin or truncated):
                ann_with_2d = dict(ann)
                ann_with_2d["bbox_2d"] = bbox_2d
                # Convert location/rotation to camera frame
                loc_world = ann["location"]
                rot_world = ann["rotation"]
                cam_loc = cam_transform.location
                cam_rot = cam_transform.rotation
                cam_yaw = math.radians(cam_rot.yaw)
                cy, sy = math.cos(cam_yaw), math.sin(cam_yaw)
                dx = loc_world["x"] - cam_loc.x
                dy = loc_world["y"] - cam_loc.y
                ann_with_2d["location"] = {
                    "x": cy*dx + sy*dy,
                    "y": -(-sy*dx + cy*dy),   # Y=left (standard)
                    "z": loc_world["z"] - cam_loc.z + ann["bbox_3d"]["z"] / 2,  # geometric center
                }
                ann_with_2d["rotation"]["roll"] = rot_world["roll"] + cam_rot.roll    # 左系相对相机
                ann_with_2d["rotation"]["pitch"] = rot_world["pitch"] - cam_rot.pitch
                ann_with_2d["rotation"]["yaw"] = -(rot_world["yaw"] - cam_rot.yaw)  # 左系, yaw取反
                bboxes_2d.append(ann_with_2d)

        with open(ann_path, "w") as f:
            json.dump(bboxes_2d, f)

    def _write_lidar_annotations(self, ann_path, channel, spec, frame, lidar_points, min_pts, ego_tf, lidar_tf):
        lidar_transform = self._lidar_sensors.get(channel)
        if lidar_transform is None or not lidar_transform.is_alive:
            return
        lidar_loc = lidar_transform.get_transform().location
        lidar_range = spec.get("output", {}).get("range", 100)
        upper_fov = math.radians(spec.get("output", {}).get("upper_fov", 10))
        lower_fov = math.radians(spec.get("output", {}).get("lower_fov", -30))
        h_fov = math.radians(spec.get("output", {}).get("horizontal_fov", 360))
        lidar_yaw = math.radians(lidar_transform.get_transform().rotation.yaw)

        actors = list(self._world.get_actors())
        annotations = []
        ego_id = self._vehicle.id

        for actor in actors:
            try:
                if not actor.is_alive or actor.id == ego_id:
                    continue
                type_id = str(actor.type_id)
                if type_id.startswith("vehicle."):
                    category = "vehicle"
                elif type_id.startswith("walker.pedestrian."):
                    category = "pedestrian"
                else:
                    continue

                if lidar_points is not None and self._count_points_in_actor_bbox(lidar_points, actor, ego_tf, lidar_tf) < min_pts:
                    continue

                # Build 8 corners of 3D bbox in world coordinates
                bbox = actor.bounding_box.extent  # half-extent
                t = actor.get_transform()
                dx, dy, dz = bbox.x, bbox.y, bbox.z  # half-extent IS the corner offset
                corners_local = np.array([
                    [ dx,  dy,  dz], [ dx,  dy, -dz], [ dx, -dy,  dz], [ dx, -dy, -dz],
                    [-dx,  dy,  dz], [-dx,  dy, -dz], [-dx, -dy,  dz], [-dx, -dy, -dz],
                ])
                yaw = math.radians(t.rotation.yaw)
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                Rz = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
                world_corners = (Rz @ corners_local.T).T + np.array([t.location.x, t.location.y, t.location.z])

                # Count corners within LiDAR FOV
                in_fov = 0
                for corner in world_corners:
                    cx, cy, cz = corner[0] - lidar_loc.x, corner[1] - lidar_loc.y, corner[2] - lidar_loc.z
                    d = math.sqrt(cx*cx + cy*cy + cz*cz)
                    if d <= lidar_range:
                        elev = math.asin(cz / d) if d > 0.01 else 0
                        if lower_fov <= elev <= upper_fov:
                            if h_fov >= math.radians(360):
                                in_fov += 1
                            else:
                                azim = math.atan2(cy, cx) - lidar_yaw
                                azim = (azim + math.pi) % (2 * math.pi) - math.pi
                                if abs(azim) <= h_fov / 2:
                                    in_fov += 1

                if in_fov < 4:
                    continue

                velocity = actor.get_velocity()
                loc = t.location
                rot = t.rotation
                # Convert to ego frame (use snapshotted ego_tf)
                eyaw = math.radians(ego_tf.rotation.yaw)
                cy, sy = math.cos(eyaw), math.sin(eyaw)
                dx = loc.x - ego_tf.location.x
                dy = loc.y - ego_tf.location.y
                annotations.append({
                    "actor_id": int(actor.id),
                    "category": category,
                    "type_id": type_id,
                    "bbox_3d": {
                        "x": float(bbox.x * 2), "y": float(bbox.y * 2), "z": float(bbox.z * 2),
                    },
                    "location": {
                        "x": cy*dx + sy*dy,   # forward in ego
                        "y": -(-sy*dx + cy*dy),   # Y=left (standard)
                        "z": float(loc.z - ego_tf.location.z + bbox.z),  # geometric center (bbox.z=half-extent)
                    },
                    "rotation": {
                        "roll": -float(rot.roll),
                        "pitch": float(rot.pitch),
                        "yaw": -float(rot.yaw - ego_tf.rotation.yaw),  # Y=左, yaw取反
                    },
                    "velocity": {
                        "x": float(velocity.x),
                        "y": float(velocity.y),
                        "z": float(velocity.z),
                    },
                })
            except RuntimeError:
                continue

        # Add static level vehicles (get_level_bbs), FOV-filtered
        eyaw = math.radians(ego_tf.rotation.yaw)
        cy, sy = math.cos(eyaw), math.sin(eyaw)
        label_map = {
            self._server.carla.CityObjectLabel.Car: "static_car",
            self._server.carla.CityObjectLabel.Truck: "static_truck",
            self._server.carla.CityObjectLabel.Bus: "static_bus",
            self._server.carla.CityObjectLabel.Train: "static_train",
            self._server.carla.CityObjectLabel.Motorcycle: "static_motorcycle",
            self._server.carla.CityObjectLabel.Bicycle: "static_bicycle",
            self._server.carla.CityObjectLabel.Pedestrians: "static_pedestrian",
        }
        for label, cat in label_map.items():
            try:
                bbs = self._world.get_level_bbs(label)
            except RuntimeError:
                continue
            for bb in bbs:
                # Skip if overlaps with any dynamic actor (both in world coords)
                bx, by = bb.location.x, bb.location.y
                overlap = False
                for actor in actors:
                    aloc = actor.get_transform().location
                    if abs(bx - aloc.x) < 3.0 and abs(by - aloc.y) < 3.0:
                        overlap = True; break
                if overlap: continue
                hx, hy, hz = bb.extent.x, bb.extent.y, bb.extent.z
                if lidar_points is not None and ego_tf is not None:
                    eyaw_s = math.radians(ego_tf.rotation.yaw)
                    ecos_s, esin_s = math.cos(eyaw_s), math.sin(eyaw_s)
                    dx_ws = bb.location.x - ego_tf.location.x
                    dy_ws = bb.location.y - ego_tf.location.y
                    cx_egos = dx_ws * ecos_s + dy_ws * esin_s
                    cy_egos = -dx_ws * esin_s + dy_ws * ecos_s
                    cy_ego_ls = -cy_egos
                    n_static = int(((np.abs(lidar_points[:, 0] - cx_egos) <= hx) & \
                                    (np.abs(lidar_points[:, 1] - cy_ego_ls) <= hy)).sum())
                    if n_static < min_pts:
                        continue
                # Build 8 corners and check FOV
                corners = np.array([
                    [hx,hy,hz],[hx,hy,-hz],[hx,-hy,hz],[hx,-hy,-hz],
                    [-hx,hy,hz],[-hx,hy,-hz],[-hx,-hy,hz],[-hx,-hy,-hz]
                ]) + np.array([bb.location.x, bb.location.y, bb.location.z])
                in_fov = 0
                for c in corners:
                    cx, cy_c, cz = c[0]-lidar_loc.x, c[1]-lidar_loc.y, c[2]-lidar_loc.z
                    d = math.sqrt(cx*cx+cy_c*cy_c+cz*cz)
                    if d <= lidar_range:
                        elev = math.asin(cz/d) if d>0.01 else 0
                        if lower_fov <= elev <= upper_fov:
                            in_fov += 1
                if in_fov < 4:
                    continue
                dx = bb.location.x - ego_tf.location.x
                dy = bb.location.y - ego_tf.location.y
                fw = cy*dx + sy*dy
                rt = -sy*dx + cy*dy
                annotations.append({
                    "actor_id": -1,
                    "category": cat,
                    "type_id": str(label).split('.')[-1],
                    "bbox_3d": {
                        "x": float(hx * 2), "y": float(hy * 2), "z": float(hz * 2),
                    },
                    "location": {
                        "x": fw, "y": -rt,
                        "z": float(bb.location.z - ego_tf.location.z),
                    },
                    "rotation": {"roll": -float(bb.rotation.roll), "pitch": float(bb.rotation.pitch),
                                 "yaw": -float(bb.rotation.yaw - ego_tf.rotation.yaw)},
                    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                })

        with open(ann_path, "w") as f:
            json.dump(annotations, f)

    def _get_camera_world_transform(self, channel):
        sensor = self._rgb_sensors.get(channel)
        if sensor and sensor.is_alive:
            return sensor.get_transform()
        return None

    def _project_bbox(self, ann, cam_transform, cam_spec, K):
        """Project 3D bounding box corners to 2D image plane."""
        if cam_transform is None:
            return None

        loc = ann["location"]
        rot = ann["rotation"]
        ext = ann["bbox_3d"]

        # 8 corners of 3D bounding box in actor local frame
        dx, dy, dz = ext["x"] / 2, ext["y"] / 2, ext["z"] / 2
        corners_local = np.array([
            [ dx,  dy,  dz], [ dx,  dy, -dz], [ dx, -dy,  dz], [ dx, -dy, -dz],
            [-dx,  dy,  dz], [-dx,  dy, -dz], [-dx, -dy,  dz], [-dx, -dy, -dz],
        ])

        # Actor local → world: apply yaw rotation + translation
        yaw = math.radians(rot["yaw"])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
        world_corners = (Rz @ corners_local.T).T + np.array([loc["x"], loc["y"], loc["z"] + ext["z"] / 2])

        # World → camera coordinates
        cam_loc = cam_transform.location
        cam_rot = cam_transform.rotation

        # Camera world-to-camera: translate then rotate
        t = np.array([cam_loc.x, cam_loc.y, cam_loc.z], dtype=np.float32)
        shifted = world_corners.astype(np.float32) - t

        # Camera rotation inverse (Z→X, X→Y in CARLA)
        # CARLA camera: X=forward, Y=right, Z=up ... actually: X=right, Y=down, Z=forward
        cy = math.cos(math.radians(-cam_rot.yaw))
        sy = math.sin(math.radians(-cam_rot.yaw))
        cp = math.cos(math.radians(-cam_rot.pitch))
        sp = math.sin(math.radians(-cam_rot.pitch))
        cr = math.cos(math.radians(-cam_rot.roll))
        sr = math.sin(math.radians(-cam_rot.roll))

        # Combined rotation (yaw → pitch → roll)
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R = R_roll @ R_pitch @ R_yaw
        cam_points = (R @ shifted.T).T

        # CARLA camera axes: X=forward, Y=right, Z=up
        # Image plane: u = fx * (Y/X) + cx, v = fy * (Z/X) + cy ... no wait
        # Actually CARLA image: u goes right (Y axis), v goes down (-Z axis)
        # Project: divide by X (depth)
        X = cam_points[:, 0]  # depth axis
        Y = cam_points[:, 1]  # right axis
        Z = cam_points[:, 2]  # up axis

        # Points in front of camera
        valid = X > 0.1
        if valid.sum() < 1:
            return None

        fx, fy = K["fx"], K["fy"]
        cx, cy = K["cx"], K["cy"]
        w, h = K["width"], K["height"]

        u = fx * (Y / X) + cx
        v = fy * (-Z / X) + cy

        # Center of bbox must be in front of camera and not too far outside image
        center_X = cam_points[0, 0]  # first corner is before Rz rotation... no, all 8 are corners
        # Compute center: average of 8 corners in camera space
        center_cam = cam_points.mean(axis=0)
        cX, cY, cZ = center_cam[0], center_cam[1], center_cam[2]
        if cX <= 0.1:
            return None
        cu = fx * (cY / cX) + cx
        cv = fy * (-cZ / cX) + cy
        if cu < -w or cu > 2 * w or cv < -h or cv > 2 * h:
            return None

        # At least one corner should be in or near the image
        in_view = (u >= -w * 0.5) & (u < w * 1.5) & (v >= -h * 0.5) & (v < h * 1.5)
        in_front = valid
        if (in_view & in_front).sum() < 1:
            return None

        u_clipped = np.clip(u[valid], 0, w - 1)
        v_clipped = np.clip(v[valid], 0, h - 1)

        return [int(np.min(u_clipped)), int(np.min(v_clipped)),
                int(np.max(u_clipped)), int(np.max(v_clipped))]

    def _spawn_vehicle(self, world, bp_lib):
        model = self._config.get("vehicle", {}).get("model", "model3")
        vehicle_bp = bp_lib.find(f"vehicle.tesla.{model}")
        if not vehicle_bp:
            vehicle_bp = bp_lib.filter("vehicle.*")[0]

        spawn_points = world.get_map().get_spawn_points()
        for pt in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, pt)
            if vehicle:
                return vehicle
        raise RuntimeError("No free spawn points")

    def _wait_for_motion(self, world, vehicle, col, start_loc):
        speed_ms = col.get("motion_speed_kmh", 5) / 3.6
        required = col.get("motion_consecutive_ticks", 3)
        timeout_ticks = col.get("motion_timeout_ticks", 500)
        consecutive = 0

        for _ in range(timeout_ticks):
            world.tick()
            velocity = vehicle.get_velocity()
            planar = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
            if planar >= speed_ms:
                consecutive += 1
                if consecutive >= required:
                    return
            else:
                consecutive = 0

        raise RuntimeError(
            f"Vehicle did not reach {speed_ms*3.6:.0f}km/h within {timeout_ticks} ticks"
        )

    def _spawn_npc_vehicles(self, world, bp_lib, tm, ego_vehicle):
        traffic_cfg = self._config.get("traffic", {})
        target = traffic_cfg.get("vehicle_count", 10)
        # Log all spawned NPC blueprints
        min_count = traffic_cfg.get("min_vehicle_count", 0)
        seed = traffic_cfg.get("npc_seed", 42)

        if target == 0:
            return []

        vehicle_bps = list(bp_lib.filter("vehicle.*"))
        spawn_points = list(world.get_map().get_spawn_points())
        ego_loc = ego_vehicle.get_transform().location
        candidates = []
        for sp in spawn_points:
            if sp.location.distance(ego_loc) >= 3.0:
                candidates.append(sp)

        rng = random.Random(seed)
        created = []
        retries = 0
        while retries < 3 and len(created) < target:
            rng.shuffle(candidates)
            for sp in candidates:
                if len(created) >= target:
                    break
                bp = rng.choice(vehicle_bps)
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", "autopilot")
                actor = world.try_spawn_actor(bp, sp)
                if actor is None:
                    continue
                actor.set_autopilot(True, tm.get_port())
                created.append(actor)
            retries += 1

        if len(created) < min_count:
            for v in created:
                v.destroy()
            raise RuntimeError(
                f"Only spawned {len(created)} NPC vehicles, need at least {min_count}"
            )
        return created

    def _record_ego_pose(self, frame):
        ego_t = self._vehicle.get_transform()
        row = [frame, ego_t.location.x, -ego_t.location.y, ego_t.location.z,
               -ego_t.rotation.roll, ego_t.rotation.pitch, -ego_t.rotation.yaw]
        for spec in self._camera_specs:
            sensor = self._rgb_sensors.get(spec["channel"])
            if sensor and sensor.is_alive:
                ct = sensor.get_transform()
                row += [ct.location.x, -ct.location.y, ct.location.z,
                        -ct.rotation.roll, ct.rotation.pitch, -ct.rotation.yaw]
        self._ego_writer.writerow(row)

    def _convert_npy_to_jpg(self):
        from tools.npy2jpg import convert_run
        convert_run(self._output_dir)

    def _make_output_dir(self):
        base = self._config.get("collection", {}).get("output_dir", "output")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(base, run_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _sigint_handler(self, signum, frame):
        print("\nInterrupted, shutting down...")
        self._running = False

    def _cleanup(self):
        print("Cleaning up...")
        for name, sensor in self._sensors.items():
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        self._sensors.clear()

        for v in self._npc_vehicles:
            if v.is_alive:
                v.destroy()
        self._npc_vehicles.clear()

        if self._vehicle and self._vehicle.is_alive:
            self._vehicle.destroy()

        if hasattr(self, '_ego_f') and self._ego_f and not self._ego_f.closed:
            self._ego_f.close()

        self._server.shutdown()
        print("Done.")
