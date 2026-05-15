import math
import numpy as np

# CARLA semantic tag → OCC voxel label (0=free, others=original CARLA tag)
OCC_CATEGORIES = {
    0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building", 4: "wall",
    5: "fence", 6: "pole", 7: "traffic_light", 8: "traffic_sign",
    9: "vegetation", 10: "terrain", 11: "sky", 12: "pedestrian",
    13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train",
    18: "motorcycle", 19: "bicycle", 20: "static", 21: "dynamic",
    22: "other", 23: "water", 24: "road_line", 25: "ground",
    26: "bridge", 27: "rail_track", 28: "guard_rail",
}


def build_occupancy_grid(ego_vehicle, annotations, settings, camera_points=None):
    """Build 3D voxel occupancy grid from camera back-projected points."""
    x_min = settings.get("x_min_m", -20.0)
    x_max = settings.get("x_max_m", 80.0)
    y_min = settings.get("y_min_m", -40.0)
    y_max = settings.get("y_max_m", 40.0)
    z_min = settings.get("z_min_m", -2.0)
    z_max = settings.get("z_max_m", 4.0)
    resolution = settings.get("resolution_m", 0.2)

    nx = int(round((x_max - x_min) / resolution))
    ny = int(round((y_max - y_min) / resolution))
    nz = int(round((z_max - z_min) / resolution))
    grid = np.zeros((nz, ny, nx), dtype=np.uint8)

    # --- Voxelize camera back-projected points (environment) ---
    if camera_points is not None and len(camera_points) > 0:
        _voxelize_lidar(grid, camera_points, ego_vehicle,
                        x_min, y_min, z_min, resolution)

    # --- Rasterize actor bboxes (dynamic objects, overrides LiDAR) ---
    ego_transform = ego_vehicle.get_transform()
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    ego_z = ego_transform.location.z
    ego_yaw = math.radians(ego_transform.rotation.yaw)
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)

    # Actor→OCC category mapping
    actor_cat_map = {"vehicle": 14, "pedestrian": 12}  # car, pedestrian

    for ann in annotations:
        cat_id = actor_cat_map.get(ann.get("category"), 21)  # default: dynamic

        ax = ann["location"]["x"]
        ay = ann["location"]["y"]
        az = ann["location"]["z"]
        actor_yaw = math.radians(ann["rotation"]["yaw"])
        rel_yaw = actor_yaw - ego_yaw

        dx, dy, dz = ax - ego_x, ay - ego_y, az - ego_z
        cx = cos_yaw * dx + sin_yaw * dy
        cy = -sin_yaw * dx + cos_yaw * dy
        cz = dz

        half_x = ann["bbox_3d"]["x"] / 2
        half_y = ann["bbox_3d"]["y"] / 2
        half_z = ann["bbox_3d"]["z"] / 2

        cos_rel = math.cos(rel_yaw)
        sin_rel = math.sin(rel_yaw)
        local_xy = np.array(
            [[half_x, half_y], [half_x, -half_y], [-half_x, -half_y], [-half_x, half_y]],
            dtype=np.float32,
        )
        rotation = np.array([[cos_rel, -sin_rel], [sin_rel, cos_rel]], dtype=np.float32)
        corners_xy = np.dot(local_xy, rotation.T)
        corners_xy[:, 0] += cx
        corners_xy[:, 1] += cy

        z_low = cz - half_z
        z_high = cz + half_z

        _rasterize_voxel_bbox(grid, corners_xy, z_low, z_high, cat_id,
                              x_min, y_min, z_min, resolution)

    return grid


def _voxelize_lidar(grid, points, ego_vehicle, x_min, y_min, z_min, resolution):
    """Project LiDAR points into ego-frame voxel grid (direct write)."""
    nz, ny, nx = grid.shape

    ego_transform = ego_vehicle.get_transform()
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    ego_z = ego_transform.location.z
    ego_yaw = math.radians(ego_transform.rotation.yaw)
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)

    px = points[:, 0] - ego_x
    py = points[:, 1] - ego_y
    pz = points[:, 2] - ego_z

    if points.shape[1] >= 5:
        tags = points[:, 4].astype(np.int32)
        if np.all(tags == 0):
            tags = np.full(len(points), 20, dtype=np.int32)
    else:
        tags = np.full(len(points), 20, dtype=np.int32)

    ex = cos_yaw * px + sin_yaw * py
    ey = -sin_yaw * px + cos_yaw * py
    ez = pz

    ix = np.floor((ex - x_min) / resolution).astype(np.int32)
    iy = np.floor((ey - y_min) / resolution).astype(np.int32)
    iz = np.floor((ez - z_min) / resolution).astype(np.int32)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    ix, iy, iz, tags = ix[valid], iy[valid], iz[valid], tags[valid]

    for i in range(len(ix)):
        tag = tags[i]
        if 0 < tag < 256:
            grid[iz[i], iy[i], ix[i]] = tag


def _rasterize_voxel_bbox(grid, corners_xy, z_low, z_high, cat_id,
                          x_min, y_min, z_min, resolution):
    nz, ny, nx = grid.shape

    min_x = float(np.min(corners_xy[:, 0]))
    max_x = float(np.max(corners_xy[:, 0]))
    min_y = float(np.min(corners_xy[:, 1]))
    max_y = float(np.max(corners_xy[:, 1]))

    ix_start = max(0, int(math.floor((min_x - x_min) / resolution)))
    ix_end = min(nx - 1, int(math.floor((max_x - x_min) / resolution)))
    iy_start = max(0, int(math.floor((min_y - y_min) / resolution)))
    iy_end = min(ny - 1, int(math.floor((max_y - y_min) / resolution)))
    iz_start = max(0, int(math.floor((z_low - z_min) / resolution)))
    iz_end = min(nz - 1, int(math.floor((z_high - z_min) / resolution)))

    if ix_end < ix_start or iy_end < iy_start or iz_end < iz_start:
        return

    for iz in range(iz_start, iz_end + 1):
        for iy in range(iy_start, iy_end + 1):
            for ix in range(ix_start, ix_end + 1):
                cx = x_min + (ix + 0.5) * resolution
                cy = y_min + (iy + 0.5) * resolution
                if _point_in_quad(cx, cy, corners_xy):
                    grid[iz, iy, ix] = cat_id


def _point_in_quad(x, y, corners):
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        if (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) > 0:
            return False
    return True
