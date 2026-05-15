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

    def run(self):
        self._server.connect()
        carla = self._server.carla
        client = self._server.client

        town = self._config["carla"].get("town")
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
        print(f"Saving data to {self._output_dir}")

        self._ego_csv = os.path.join(self._output_dir, "ego_trajectory.csv")
        self._ego_f = open(self._ego_csv, "w", newline="")
        import csv as csv_module
        self._ego_writer = csv_module.writer(self._ego_f)
        self._ego_writer.writerow(["frame", "x", "y", "z", "roll", "pitch", "yaw",
                                    "cam_x", "cam_y", "cam_z", "cam_roll", "cam_pitch", "cam_yaw"])

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
        self._init_annotations()
        print(f"Attached sensors: {', '.join(self._sensors.keys())}")

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

                if duration > 0 and (time.time() - start_time) >= duration:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted, shutting down...")
        finally:
            self._cleanup()

        self._convert_npy_to_jpg()

    def _build_sensor_layout(self, world_fps):
        layout = self._config.get("_sensor_layout", {})
        sensors = list(layout.get("sensors", []))
        gt = self._config.get("ground_truth", {})

        # Auto-tag sensors with subdir for grouped output (skip pseudo-sensors)
        for s in sensors:
            if s.get("modality") in ("camera_rgb", "lidar", "lidar_semantic"):
                s.setdefault("subdir", "original")

        if gt.get("depth", True):
            for s in layout.get("sensors", []):
                if s.get("modality") == "camera_rgb" and s.get("enabled", True):
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

        if gt.get("semantic_camera", True):
            for s in layout.get("sensors", []):
                if s.get("modality") == "camera_rgb" and s.get("enabled", True):
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

        if self._occ_spec is not None:
            from .sensors import _sensor_output_dir
            self._occ_save_dir = _sensor_output_dir(self._output_dir, self._occ_spec)
            # Save OCC grid params for post-processing
            import json as _json
            occ_cfg = self._occ_spec.get("output", {})
            with open(os.path.join(self._occ_save_dir, "grid_config.json"), "w") as _f:
                _json.dump(occ_cfg, _f)
            self._occ_save_dir = None  # OCC generated in post-processing

    def _write_annotations_for_new_frames(self):
        from .sensors import get_sensor_frames

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
            self._write_annotations(frame, spec, channel, ann_path)

        for spec in self._lidar_specs:
            channel = spec["channel"]
            frame = get_sensor_frames().get(channel)
            if frame is None:
                continue
            ann_dir = os.path.join(self._output_dir, channel, "annotations")
            ann_path = os.path.join(ann_dir, f"{frame:08d}.json")
            if os.path.exists(ann_path):
                continue
            self._write_lidar_annotations(ann_path, channel, spec)

    def _write_annotations(self, frame, spec, channel, ann_path):
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

                bbox = actor.bounding_box.extent
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                annotations.append({
                    "actor_id": int(actor.id),
                    "category": category,
                    "type_id": type_id,
                    "bbox_3d": {
                        "x": float(bbox.x), "y": float(bbox.y), "z": float(bbox.z),
                    },
                    "location": {
                        "x": float(transform.location.x),
                        "y": float(transform.location.y),
                        "z": float(transform.location.z),
                    },
                    "rotation": {
                        "roll": float(transform.rotation.roll),
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

        cam_transform = self._get_camera_world_transform(channel)
        K = self._frame_annotations[channel]["intrinsic"]

        bboxes_2d = []
        for ann in annotations:
            bbox_2d = self._project_bbox(ann, cam_transform, spec, K)
            if bbox_2d is not None:
                ann_with_2d = dict(ann)
                ann_with_2d["bbox_2d"] = bbox_2d
                bboxes_2d.append(ann_with_2d)

        with open(ann_path, "w") as f:
            json.dump(bboxes_2d, f)

    def _write_lidar_annotations(self, ann_path, channel, spec):
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

                # Build 8 corners of 3D bbox in world coordinates
                bbox = actor.bounding_box.extent
                t = actor.get_transform()
                dx, dy, dz = bbox.x / 2, bbox.y / 2, bbox.z / 2
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
                annotations.append({
                    "actor_id": int(actor.id),
                    "category": category,
                    "type_id": type_id,
                    "bbox_3d": {
                        "x": float(bbox.x), "y": float(bbox.y), "z": float(bbox.z),
                    },
                    "location": {
                        "x": float(t.location.x),
                        "y": float(t.location.y),
                        "z": float(t.location.z),
                    },
                    "rotation": {
                        "roll": float(t.rotation.roll),
                        "pitch": float(t.rotation.pitch),
                        "yaw": float(t.rotation.yaw),
                    },
                    "velocity": {
                        "x": float(velocity.x),
                        "y": float(velocity.y),
                        "z": float(velocity.z),
                    },
                })
            except RuntimeError:
                continue

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
        world_corners = (Rz @ corners_local.T).T + np.array([loc["x"], loc["y"], loc["z"]])

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

    def _backproject_camera_points(self, frame):
        """Back-project depth+semantic pixels to 3D points in world coordinates."""
        all_points = []
        for spec in self._camera_specs:
            channel = spec["channel"]
            # Find nearest depth frame
            depth_dir = os.path.join(self._output_dir, channel, "depth")
            if not os.path.isdir(depth_dir):
                continue
            depth_files = sorted(os.listdir(depth_dir))
            if not depth_files:
                continue
            depth_frames = [int(f.replace(".npy", "")) for f in depth_files if f.endswith(".npy")]
            nearest_df = min(depth_frames, key=lambda x: abs(x - frame))
            if abs(nearest_df - frame) > 10:
                continue
            depth = np.load(os.path.join(depth_dir, f"{nearest_df:08d}.npy"))

            # Find nearest semantic frame
            sem_dir = os.path.join(self._output_dir, channel, "semantic")
            tags = None
            if os.path.isdir(sem_dir):
                sem_files = sorted(os.listdir(sem_dir))
                sem_frames = [int(f.replace(".png", "")) for f in sem_files if f.endswith(".png")]
                if sem_frames:
                    nearest_sf = min(sem_frames, key=lambda x: abs(x - frame))
                    if abs(nearest_sf - frame) <= 10:
                        tags = np.array(Image.open(os.path.join(sem_dir, f"{nearest_sf:08d}.png")))

            h, w = depth.shape
            # Downsample: every 4th pixel → 16x faster, 90k points per frame
            step = 4
            depth = depth[::step, ::step]
            if tags is not None:
                tags = tags[::step, ::step]
            h, w = depth.shape
            # Camera intrinsic (adjusted for downsampled image)
            from .sensors import _compute_intrinsic
            K = _compute_intrinsic(spec)
            fx, fy = K["fx"] / step, K["fy"] / step
            cx, cy = K["cx"] / step, K["cy"] / step

            # Pixel coordinates for downsampled image
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            # Back-project: X_cam = Z, Y_cam = (u-cx)*Z/fx, Z_cam = -(v-cy)*Z/fy
            Z = depth  # forward depth
            X_cam = (u - cx) * Z / fx   # right in camera frame
            Y_cam = -(v - cy) * Z / fy  # up in camera frame... wait, CARLA camera: X=forward, Y=right, Z=up
            # In CARLA: image u → right (Y axis), v → down (-Z axis)
            # So: forward=X_cam=Z_depth, right=Y_cam=(u-cx)*Z/fx, up=Z_cam=-(v-cy)*Z/fy

            # Filter invalid depths
            valid = (Z > 0.1) & (Z < 999)
            if valid.sum() == 0:
                continue

            # Camera coord: X=forward, Y=right, Z=up
            pts_cam = np.stack([
                Z[valid],                              # forward (X in CARLA)
                (u[valid] - cx) * Z[valid] / fx,      # right (Y in CARLA)
                -(v[valid] - cy) * Z[valid] / fy,     # up (Z in CARLA)
            ], axis=-1)  # (N, 3)

            # Transform from camera to world
            cam_sensor = self._rgb_sensors.get(channel)
            if cam_sensor is None or not cam_sensor.is_alive:
                continue
            cam_t = cam_sensor.get_transform()
            cam_loc = cam_t.location
            cam_rot = cam_t.rotation

            # Camera world transform: first rotate, then translate
            cy = math.cos(math.radians(cam_rot.yaw))
            sy = math.sin(math.radians(cam_rot.yaw))
            cp = math.cos(math.radians(cam_rot.pitch))
            sp = math.sin(math.radians(cam_rot.pitch))
            cr = math.cos(math.radians(cam_rot.roll))
            sr = math.sin(math.radians(cam_rot.roll))

            R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            R = R_yaw @ R_pitch @ R_roll  # camera→world rotation

            # Camera coords → world coords
            pts_world = (R @ pts_cam.T).T + np.array([cam_loc.x, cam_loc.y, cam_loc.z])

            # Attach semantic tags
            if tags is not None:
                tag_values = tags[valid].reshape(-1, 1)
            else:
                tag_values = np.full((len(pts_world), 1), 20)  # static

            pts_with_tag = np.hstack([pts_world, np.zeros((len(pts_world), 1)), tag_values, np.zeros((len(pts_world), 1))])
            all_points.append(pts_with_tag)

        if all_points:
            return np.vstack(all_points)
        return None

    def _record_ego_pose(self, frame):
        ego_t = self._vehicle.get_transform()
        row = [frame, ego_t.location.x, ego_t.location.y, ego_t.location.z,
               ego_t.rotation.roll, ego_t.rotation.pitch, ego_t.rotation.yaw]
        for spec in self._camera_specs:
            sensor = self._rgb_sensors.get(spec["channel"])
            if sensor and sensor.is_alive:
                ct = sensor.get_transform()
                row += [ct.location.x, ct.location.y, ct.location.z,
                        ct.rotation.roll, ct.rotation.pitch, ct.rotation.yaw]
        self._ego_writer.writerow(row)

    def _convert_npy_to_jpg(self):
        from tools.npy2jpg import convert_run
        convert_run(self._output_dir)

    def _write_occ(self, frame):
        from .occupancy import build_occupancy_grid

        # Fuse depth + semantic camera into dense 3D points
        camera_points = self._backproject_camera_points(frame)

        lidar_points = None

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
                bbox = actor.bounding_box.extent
                t = actor.get_transform()
                annotations.append({
                    "category": category,
                    "bbox_3d": {
                        "x": float(bbox.x), "y": float(bbox.y), "z": float(bbox.z),
                    },
                    "location": {
                        "x": float(t.location.x),
                        "y": float(t.location.y),
                        "z": float(t.location.z),
                    },
                    "rotation": {
                        "yaw": float(t.rotation.yaw),
                    },
                })
            except RuntimeError:
                continue

        occ_cfg = self._occ_spec.get("output", {})
        grid = build_occupancy_grid(self._vehicle, annotations, occ_cfg, camera_points)
        np.save(os.path.join(self._occ_save_dir, f"{frame:08d}.npy"), grid)
        # Save metadata once
        meta_path = os.path.join(self._occ_save_dir, "metadata.json")
        if not os.path.exists(meta_path):
            import json
            with open(meta_path, "w") as f:
                json.dump({
                    "x_min_m": occ_cfg.get("x_min_m", -20),
                    "x_max_m": occ_cfg.get("x_max_m", 80),
                    "y_min_m": occ_cfg.get("y_min_m", -40),
                    "y_max_m": occ_cfg.get("y_max_m", 40),
                    "z_min_m": occ_cfg.get("z_min_m", -2),
                    "z_max_m": occ_cfg.get("z_max_m", 4),
                    "resolution_m": occ_cfg.get("resolution_m", 0.2),
                    "categories": {
                        "0": "free", "1": "road", "2": "sidewalk", "3": "building",
                        "4": "wall", "5": "fence", "6": "pole", "7": "traffic_light",
                        "8": "traffic_sign", "9": "vegetation", "10": "terrain",
                        "11": "sky", "12": "pedestrian", "13": "rider", "14": "car",
                        "15": "truck", "16": "bus", "17": "train", "18": "motorcycle",
                        "19": "bicycle", "20": "static", "21": "dynamic", "22": "other",
                        "23": "water", "24": "road_line", "25": "ground",
                        "26": "bridge", "27": "rail_track", "28": "guard_rail",
                    },
                }, f)

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
