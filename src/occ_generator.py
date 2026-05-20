"""
Semantic Occupancy Ground Truth generator using CARLA GT Geometry.

Strategy (from occ生成策略.md):
  - Static OCC built once per map from waypoints (road/sidewalk) + get_level_bbs
  - Per-frame OCC = crop static_occ to ego-centric + overlay dynamic actors
  - Labels follow README CARLA semantic tags (0-28)
"""

import math
import numpy as np

# ── OCC space (ego-centric, left-hand: X=forward, Y=left, Z=up) ──
PC_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
VOXEL_SIZE = [0.5, 0.5, 0.5]
OCC_SHAPE = [200, 200, 16]  # X, Y, Z


def _clamp_voxel_indices(x_min, x_max, y_min, y_max, z_min, z_max,
                         pc_range, voxel_size, shape):
    """Convert world AABB to voxel index range, clamped to grid bounds."""
    ix0 = int(math.floor((x_min - pc_range[0]) / voxel_size[0]))
    ix1 = int(math.floor((x_max - pc_range[0]) / voxel_size[0]))
    iy0 = int(math.floor((y_min - pc_range[1]) / voxel_size[1]))
    iy1 = int(math.floor((y_max - pc_range[1]) / voxel_size[1]))
    iz0 = int(math.floor((z_min - pc_range[2]) / voxel_size[2]))
    iz1 = int(math.floor((z_max - pc_range[2]) / voxel_size[2]))

    ix0 = max(0, ix0)
    ix1 = min(shape[0] - 1, ix1)
    iy0 = max(0, iy0)
    iy1 = min(shape[1] - 1, iy1)
    iz0 = max(0, iz0)
    iz1 = min(shape[2] - 1, iz1)

    if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
        return None
    return ix0, ix1, iy0, iy1, iz0, iz1


# ── Dynamic actor type → CARLA semantic tag ──

def _actor_semantic_tag(type_id):
    """Map CARLA actor type_id or static.* type_id to semantic tag."""
    type_str = str(type_id)
    # Dynamic actors
    if type_str.startswith("vehicle."):
        for prefix, tag in [("vehicle.car", 14), ("vehicle.truck", 15),
                            ("vehicle.bus", 16), ("vehicle.train", 17),
                            ("vehicle.motorcycle", 18), ("vehicle.bicycle", 19)]:
            if type_str.startswith(prefix):
                return tag
        return 14  # generic vehicle → car
    if type_str.startswith("walker.pedestrian"):
        return 12
    if type_str.startswith("walker"):
        return 12
    # Static level_bbs (same categories as camera annotations)
    static_map = {
        "static.car": 14, "static.truck": 15, "static.bus": 16,
        "static.train": 17, "static.motorcycle": 18, "static.bicycle": 19,
        "static.pedestrian": 12,
    }
    if type_str in static_map:
        return static_map[type_str]
    return 21  # dynamic (fallback)


# ── CARLA CityObjectLabel → semantic tag ──
STATIC_CITY_OBJECT_TAG = {
    "Buildings": 3,
    "Walls": 4,
    "Fences": 5,
    "Poles": 6,
    "TrafficSigns": 8,
    "Vegetation": 9,
    "Other": 22,
    "GuardRails": 28,
    "RoadLines": 24,
    "Bridges": 26,
    "RailTracks": 27,
}


def _get_static_object_tag(obj_label):
    name = str(obj_label).split('.')[-1]
    return STATIC_CITY_OBJECT_TAG.get(name, 22)


# ── Static OCC building ──

def fill_road_from_waypoints(occ, world, pc_range=None, voxel_size=None,
                             waypoint_gap=0.5, z_thickness=0.2):
    """Fill road (tag 1) and sidewalk (tag 2) voxels from waypoints.

    Uses voxel-index range filling per waypoint for speed.
    """
    if pc_range is None:
        pc_range = PC_RANGE
    if voxel_size is None:
        voxel_size = VOXEL_SIZE

    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(waypoint_gap)

    for wp in waypoints:
        try:
            import carla
            if wp.lane_type == carla.LaneType.Driving:
                tag = 1
            elif wp.lane_type == carla.LaneType.Sidewalk:
                tag = 2
            else:
                continue
        except (ImportError, NameError):
            if int(wp.lane_type) == 1:
                tag = 1
            elif int(wp.lane_type) == 2:
                tag = 2
            else:
                continue

        loc = wp.transform.location
        yaw = math.radians(wp.transform.rotation.yaw)
        lane_width = wp.lane_width
        half_w = lane_width / 2.0

        # Perpendicular direction (right vector in world coords)
        rx = math.sin(yaw)
        ry = -math.cos(yaw)

        # Compute world AABB for this waypoint segment
        dx = abs(rx * half_w)
        dy = abs(ry * half_w)
        x_min_w = loc.x - dx
        x_max_w = loc.x + dx
        y_min_w = loc.y - dy
        y_max_w = loc.y + dy
        z_min_w = loc.z - z_thickness
        z_max_w = loc.z + z_thickness

        indices = _clamp_voxel_indices(x_min_w, x_max_w, y_min_w, y_max_w,
                                       z_min_w, z_max_w, pc_range, voxel_size, occ.shape)
        if indices is None:
            continue
        ix0, ix1, iy0, iy1, iz0, iz1 = indices

        # Only fill voxels that are currently 0 (unlabeled)
        slab = occ[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]
        slab[slab == 0] = tag


def fill_static_bbox(occ, bbox, tag, pc_range=None, voxel_size=None):
    """Fill voxels inside a CARLA level_bbs BoundingBox (AABB in world frame)."""
    if pc_range is None:
        pc_range = PC_RANGE
    if voxel_size is None:
        voxel_size = VOXEL_SIZE

    try:
        import carla
        verts = bbox.get_world_vertices(carla.Transform())
    except (ImportError, NameError):
        import carla
        verts = bbox.get_world_vertices(carla.Transform())

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]

    indices = _clamp_voxel_indices(min(xs), max(xs), min(ys), max(ys),
                                   min(zs), max(zs), pc_range, voxel_size, occ.shape)
    if indices is None:
        return
    ix0, ix1, iy0, iy1, iz0, iz1 = indices
    occ[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = tag


def build_static_occ(world, pc_range=None, voxel_size=None, occ_shape=None):
    """Build full-map static occupancy grid (one-time operation per map)."""
    if pc_range is None:
        pc_range = PC_RANGE
    if voxel_size is None:
        voxel_size = VOXEL_SIZE
    if occ_shape is None:
        occ_shape = OCC_SHAPE

    occ = np.zeros(occ_shape, dtype=np.uint8)

    print("  Filling road/sidewalk from waypoints...")
    fill_road_from_waypoints(occ, world, pc_range, voxel_size)

    try:
        import carla
        city_labels = [
            carla.CityObjectLabel.Buildings,
            carla.CityObjectLabel.Walls,
            carla.CityObjectLabel.Fences,
            carla.CityObjectLabel.Poles,
            carla.CityObjectLabel.TrafficSigns,
            carla.CityObjectLabel.Vegetation,
        ]
    except ImportError:
        city_labels = []

    for obj_label in city_labels:
        tag = _get_static_object_tag(obj_label)
        try:
            bbs = world.get_level_bbs(obj_label)
        except RuntimeError:
            continue
        print(f"  Filling {len(bbs)} {str(obj_label).split('.')[-1]} (tag={tag})...")
        for bb in bbs:
            fill_static_bbox(occ, bb, tag, pc_range, voxel_size)

    return occ


# ── Per-frame OCC generation ──

def build_frame_occ(static_occ, static_pc_range, ego_transform, dynamic_actors,
                    pc_range=None, voxel_size=None, occ_shape=None):
    """Build per-frame ego-centric OCC from static_occ + dynamic actors.

    Args:
        static_occ: pre-built static occupancy grid (world-aligned)
        static_pc_range: [x_min, y_min, z_min, x_max, y_max, z_max] for static_occ
        ego_transform: carla.Transform, ego vehicle world pose (CARLA coords)
        dynamic_actors: list of (transform, extent, type_id) in CARLA world coords
    Returns:
        occ: np.ndarray of shape occ_shape, dtype uint8
    """
    if pc_range is None:
        pc_range = PC_RANGE
    if voxel_size is None:
        voxel_size = VOXEL_SIZE
    if occ_shape is None:
        occ_shape = OCC_SHAPE

    ex = ego_transform.location.x
    ey = ego_transform.location.y
    ez = ego_transform.location.z
    eyaw = math.radians(ego_transform.rotation.yaw)
    cos_yaw = math.cos(eyaw)
    sin_yaw = math.sin(eyaw)

    # Vectorized crop: ego-centric → world → static_occ index
    ix_ary = np.arange(occ_shape[0], dtype=np.int32)
    iy_ary = np.arange(occ_shape[1], dtype=np.int32)
    iz_ary = np.arange(occ_shape[2], dtype=np.int32)
    ix_grid, iy_grid, iz_grid = np.meshgrid(ix_ary, iy_ary, iz_ary, indexing='ij')

    ex_local = pc_range[0] + (ix_grid + 0.5) * voxel_size[0]
    ey_local = pc_range[1] + (iy_grid + 0.5) * voxel_size[1]
    ez_local = pc_range[2] + (iz_grid + 0.5) * voxel_size[2]

    # Ego (fwd=X, left=Y) → world (fwd=X, right=Y)
    wx = ex + ex_local * cos_yaw + ey_local * sin_yaw
    wy = ey + ex_local * sin_yaw - ey_local * cos_yaw
    wz = ez + ez_local

    sx = np.floor((wx - static_pc_range[0]) / voxel_size[0]).astype(np.int32)
    sy = np.floor((wy - static_pc_range[1]) / voxel_size[1]).astype(np.int32)
    sz = np.floor((wz - static_pc_range[2]) / voxel_size[2]).astype(np.int32)

    sshape = static_occ.shape
    valid = (sx >= 0) & (sx < sshape[0]) & (sy >= 0) & (sy < sshape[1]) & (sz >= 0) & (sz < sshape[2])

    occ = np.zeros(occ_shape, dtype=np.uint8)
    occ[valid] = static_occ[sx[valid], sy[valid], sz[valid]]

    # Overlay dynamic actors
    for actor_transform, extent, type_id in dynamic_actors:
        tag = _actor_semantic_tag(type_id)
        _fill_actor_ego_centric(occ, actor_transform, extent, tag,
                                ego_transform, pc_range, voxel_size)

    return occ


def _fill_actor_ego_centric(occ, actor_transform, extent, tag,
                            ego_transform, pc_range, voxel_size):
    """Fill a dynamic actor's oriented bbox into the ego-centric OCC grid.

    Transforms voxel centers to actor local frame for precise oriented filling.
    """
    try:
        import carla
        bbox = carla.BoundingBox(carla.Location(0, 0, 0), extent)
        verts = bbox.get_world_vertices(actor_transform)
    except (ImportError, NameError):
        import carla
        bbox = carla.BoundingBox(carla.Location(0, 0, 0), extent)
        verts = bbox.get_world_vertices(actor_transform)

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    ex = ego_transform.location.x
    ey = ego_transform.location.y
    ez = ego_transform.location.z
    eyaw = math.radians(ego_transform.rotation.yaw)
    cos_yaw = math.cos(eyaw)
    sin_yaw = math.sin(eyaw)

    # World AABB → ego-centric voxel index range
    def _world_to_ego_pt(wx, wy, wz):
        dx = wx - ex
        dy = wy - ey
        fwd = dx * cos_yaw + dy * sin_yaw
        lft = dx * sin_yaw - dy * cos_yaw
        return fwd, lft, wz - ez

    # Get ego-centric AABB corners
    world_corners = [(x_min, y_min, z_min), (x_min, y_min, z_max),
                     (x_min, y_max, z_min), (x_min, y_max, z_max),
                     (x_max, y_min, z_min), (x_max, y_min, z_max),
                     (x_max, y_max, z_min), (x_max, y_max, z_max)]
    ego_corners = [_world_to_ego_pt(wx, wy, wz) for wx, wy, wz in world_corners]
    ex_min = min(c[0] for c in ego_corners)
    ex_max = max(c[0] for c in ego_corners)
    ey_min = min(c[1] for c in ego_corners)
    ey_max = max(c[1] for c in ego_corners)
    ez_min = min(c[2] for c in ego_corners)
    ez_max = max(c[2] for c in ego_corners)

    indices = _clamp_voxel_indices(ex_min, ex_max, ey_min, ey_max,
                                   ez_min, ez_max, pc_range, voxel_size, occ.shape)
    if indices is None:
        return
    ix0, ix1, iy0, iy1, iz0, iz1 = indices

    # Actor local frame: yaw rotation only (pitch/roll negligible for vehicles)
    ayaw = math.radians(actor_transform.rotation.yaw)
    acos = math.cos(-ayaw)   # inverse rotation: world → actor
    asin = math.sin(-ayaw)
    aloc = actor_transform.location
    hx, hy, hz = extent.x, extent.y, extent.z  # half-extents

    # For each voxel in the AABB, check if center is inside oriented bbox
    ix_range = np.arange(ix0, ix1 + 1)
    iy_range = np.arange(iy0, iy1 + 1)
    iz_range = np.arange(iz0, iz1 + 1)
    ix_g, iy_g, iz_g = np.meshgrid(ix_range, iy_range, iz_range, indexing='ij')

    # Voxel centers in ego frame
    vx = pc_range[0] + (ix_g + 0.5) * voxel_size[0]
    vy = pc_range[1] + (iy_g + 0.5) * voxel_size[1]
    vz = pc_range[2] + (iz_g + 0.5) * voxel_size[2]

    # Ego frame → world
    wx = ex + vx * cos_yaw - vy * sin_yaw      # Y=left → right: vx*fwd + (-vy)*right
    wy = ey + vx * sin_yaw + vy * cos_yaw      # ... wait, need to re-derive
    wz = ez + vz

    # Let me redo: ego frame (fwd=X, left=Y) → CARLA world (fwd=X, right=Y)
    # world_forward = ego_forward
    # world_right = -ego_left
    # wx = ex + vx*cos_yaw - (-vy)*sin_yaw = ex + vx*cos_yaw + vy*sin_yaw
    # wy = ey + vx*sin_yaw + (-vy)*cos_yaw = ey + vx*sin_yaw - vy*cos_yaw
    # Wait, that's wrong too. Let me re-derive carefully.
    #
    # Ego frame: a point at (fwd, left) = (vx, vy)
    # In world's forward/right frame: (vx, -vy)  [left = -right]
    # Rotate by yaw:
    #   wx = ex + vx*cos_yaw - (-vy)*sin_yaw = ex + vx*cos_yaw + vy*sin_yaw
    #   wy = ey + vx*sin_yaw + (-vy)*cos_yaw = ey + vx*sin_yaw - vy*cos_yaw
    #
    # This matches my previous correction in build_frame_occ.

    wx = ex + vx * cos_yaw + vy * sin_yaw
    wy = ey + vx * sin_yaw - vy * cos_yaw

    # World → actor local frame
    dx = wx - aloc.x
    dy = wy - aloc.y
    lx = dx * acos - dy * asin   # rotate to actor frame
    ly = dx * asin + dy * acos
    lz = wz - aloc.z

    inside = (np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(lz) <= hz)

    occ[ix_g[inside], iy_g[inside], iz_g[inside]] = tag
