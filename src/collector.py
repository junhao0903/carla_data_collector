import json
import math
import os
import random
import time
import signal
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm


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

        town = self._config["carla"].get("map")
        if town:
            self._world = client.load_world(town)
        else:
            self._world = client.get_world()

        fps = self._config["carla"].get("fps", 20)

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        self._world.apply_settings(settings)

        self._ensure_static_occ(self._world)

        bp_lib = self._world.get_blueprint_library()
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
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
        # Save metadata for post-processing
        import yaml as _yaml
        meta = dict(self._config.get("_sensor_layout", {}))
        sync = self._config.get("collection", {}).get("synchronous", True)
        meta["_collection"] = {"synchronous": sync, "fps": fps}
        with open(os.path.join(self._output_dir, "sensor_layout.yaml"), "w") as _f:
            _yaml.dump(meta, _f)
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

        duration = self._config.get("collection", {}).get("duration_seconds", 0)
        start_time = time.time()
        if duration > 0:
            h, r = divmod(duration, 3600)
            m, s = divmod(r, 60)
            total_str = f"{int(h):d}h{int(m):02d}m{int(s):02d}s" if h else f"{int(m):d}m{int(s):02d}s"
            pbar = tqdm(total=duration, desc="Collecting", unit="s",
                        bar_format="{desc}: {percentage:.0f}%|{bar}| {elapsed} / " + total_str,
                        dynamic_ncols=True)
        else:
            pbar = None
        _static_saved = False
        try:
            while True:
                from .sensors import set_world_frame
                frame = self._world.get_snapshot().frame + 1
                set_world_frame(frame)
                self._world.tick()

                if not _static_saved:
                    self._save_global_static_bboxes()
                    _static_saved = True
                self._save_dynamic_actors(frame)
                self._record_ego_pose(frame)

                if pbar is not None:
                    elapsed = time.time() - start_time
                    pbar.n = min(elapsed, duration)
                    pbar.refresh()
                if duration > 0 and (time.time() - start_time) >= duration:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted, shutting down...")
        finally:
            if pbar is not None:
                pbar.close()
            self._cleanup()

        self._run_post_processing()

    def _save_dynamic_actors(self, frame):
        """Save all dynamic actors in autonomous driving coords (X=fwd,Y=left,Z=up).

        Converts from CARLA world (Y=right) by negating Y, pitch, yaw.
        """
        ann_dir = os.path.join(self._output_dir, "ANNO", "dynamic_actors")
        os.makedirs(ann_dir, exist_ok=True)

        annotations = []
        ego_id = self._vehicle.id

        for actor in self._world.get_actors():
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

                extent = actor.bounding_box.extent
                bx, by, bz = extent.x, extent.y, extent.z
                if bx <= 0.01: bx = 2.0
                if by <= 0.01: by = 1.0
                if bz <= 0.01: bz = 0.75
                t = actor.get_transform()
                vel = actor.get_velocity()

                annotations.append({
                    "actor_id": int(actor.id),
                    "category": category,
                    "type_id": type_id,
                    "bbox_3d": {
                        "x": float(bx * 2), "y": float(by * 2), "z": float(bz * 2),
                    },
                    "location": {
                        "x": float(t.location.x),
                        "y": -float(t.location.y),
                        "z": float(t.location.z + bz),
                    },
                    "rotation": {
                        "roll": float(t.rotation.roll),
                        "pitch": -float(t.rotation.pitch),
                        "yaw": -float(t.rotation.yaw),
                    },
                    "velocity": {
                        "x": float(vel.x),
                        "y": -float(vel.y),
                        "z": float(vel.z),
                    },
                })
            except RuntimeError:
                continue

        ann_path = os.path.join(ann_dir, f"{frame:08d}.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

    def _save_global_static_bboxes(self):
        """Save all static bboxes once in global right-hand coords (X=fwd, Y=left, Z=up)."""
        carla = self._server.carla
        label_map = {
            carla.CityObjectLabel.Car: "static.car",
            carla.CityObjectLabel.Truck: "static.truck",
            carla.CityObjectLabel.Bus: "static.bus",
            carla.CityObjectLabel.Train: "static.train",
            carla.CityObjectLabel.Motorcycle: "static.motorcycle",
            carla.CityObjectLabel.Bicycle: "static.bicycle",
            carla.CityObjectLabel.Pedestrians: "static.pedestrian",
        }
        bboxes = []
        for obj_label, type_id in label_map.items():
            try:
                bbs = sorted(self._world.get_level_bbs(obj_label),
                             key=lambda b: (b.location.x, b.location.y, b.location.z))
            except RuntimeError:
                continue
            for bb in bbs:
                bboxes.append({
                    "type_id": type_id,
                    "location": {
                        "x": bb.location.x,
                        "y": -bb.location.y,
                        "z": bb.location.z,
                    },
                    "rotation": {
                        "roll": bb.rotation.roll,
                        "pitch": -bb.rotation.pitch,
                        "yaw": -bb.rotation.yaw,
                    },
                    "bbox_3d": {
                        "x": float(bb.extent.x * 2),
                        "y": float(bb.extent.y * 2),
                        "z": float(bb.extent.z * 2),
                    },
                })
        ann_dir = os.path.join(self._output_dir, "ANNO")
        os.makedirs(ann_dir, exist_ok=True)
        path = os.path.join(ann_dir, "static_bboxes.json")
        with open(path, "w") as f:
            json.dump(bboxes, f)
        print(f"Saved {len(bboxes)} static bboxes to {path}")

    def _ensure_static_occ(self, world):
        """Generate static OCC for current map, rebuild if config changed."""
        layout = self._config.get("_sensor_layout", {})
        has_occ = any(s.get("modality") == "occupancy" and s.get("enabled", True)
                      for s in layout.get("sensors", []))
        if not has_occ:
            return

        town = world.get_map().name.split("/")[-1]
        npy_path = os.path.join("map", f"{town}_static_occ.npy")
        json_path = npy_path.replace(".npy", ".json")

        spawn_pts = world.get_map().get_spawn_points()
        xs = [sp.location.x for sp in spawn_pts]
        ys = [sp.location.y for sp in spawn_pts]
        margin = 100.0
        pc_range = [min(xs) - margin, min(ys) - margin, -5.0,
                     max(xs) + margin, max(ys) + margin, 15.0]
        voxel_size = [0.5, 0.5, 0.5]
        for s in layout.get("sensors", []):
            if s.get("modality") == "occupancy" and s.get("enabled", True):
                voxel_size = s.get("output", {}).get("voxel_size", voxel_size)
                break

        if os.path.exists(npy_path) and os.path.exists(json_path):
            import json as _json
            with open(json_path) as f:
                meta = _json.load(f)
            if (meta.get("map") == town
                    and meta.get("pc_range") == pc_range
                    and meta.get("voxel_size") == voxel_size):
                return

        print(f"Building static OCC for {town} (one-time, ~1-2 min)...")
        os.makedirs("map", exist_ok=True)
        from .occ_generator import build_static_occ
        shape = [int(round((pc_range[3] - pc_range[0]) / voxel_size[0])),
                 int(round((pc_range[4] - pc_range[1]) / voxel_size[1])),
                 int(round((pc_range[5] - pc_range[2]) / voxel_size[2]))]
        occ = build_static_occ(world, pc_range, voxel_size, shape)
        np.save(npy_path, occ)
        import json as _json
        with open(json_path, "w") as f:
            _json.dump({"map": town, "pc_range": pc_range,
                         "voxel_size": voxel_size, "shape": shape}, f)
        print(f"Static OCC saved to {npy_path} ({occ.nbytes / 1e6:.0f} MB)")

    def _setup_filter_lidar(self, world, bp_lib, vehicle, cfg):
        bp = bp_lib.find(cfg.get("blueprint", "sensor.lidar.ray_cast"))
        bp.set_attribute("channels", str(cfg.get("channels", 64)))
        bp.set_attribute("range", str(cfg.get("range", 100)))
        bp.set_attribute("points_per_second", str(cfg.get("points_per_second", 56000)))
        bp.set_attribute("rotation_frequency", str(cfg.get("rotation_frequency", 10)))

        from .sensors import _make_transform, _sensor_data, _sensor_frames
        transform = _make_transform(cfg.get("transform", {}))
        sensor = world.spawn_actor(bp, transform, attach_to=vehicle)

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
            from .sensors import _compute_intrinsic
            self._frame_annotations[channel] = {
                "intrinsic": _compute_intrinsic(spec),
            }
        if self._occ_spec is not None:
            import json as _json
            occ_cfg = self._occ_spec.get("output", {})
            occ_cfg["static_occ_dir"] = occ_cfg.get("static_occ_dir", "map")
            occ_cfg["map"] = self._config["carla"].get("map", "")
            os.makedirs(os.path.join(self._output_dir, "OCC"), exist_ok=True)
            with open(os.path.join(self._output_dir, "OCC", "occ_metadata.json"), "w") as _f:
                _json.dump(occ_cfg, _f)

    def _record_ego_pose(self, frame):
        ego_t = self._vehicle.get_transform()
        row = [frame,
               ego_t.location.x, -ego_t.location.y, ego_t.location.z,
               ego_t.rotation.roll, -ego_t.rotation.pitch, -ego_t.rotation.yaw]
        for spec in self._camera_specs:
            sensor = self._rgb_sensors.get(spec["channel"])
            if sensor and sensor.is_alive:
                ct = sensor.get_transform()
                row += [ct.location.x, -ct.location.y, ct.location.z,
                        ct.rotation.roll, -ct.rotation.pitch, -ct.rotation.yaw]
        self._ego_writer.writerow(row)

    def _run_post_processing(self):
        from tools.post_process import post_process
        from tools.visualize import convert_run
        post_process(self._output_dir)
        convert_run(self._output_dir)

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
