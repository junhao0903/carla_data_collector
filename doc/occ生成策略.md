# CARLA Semantic Occupancy（Occ）完整生成方案（工程可落地版）

# 1. 目标

当前目标是在 CARLA 中生成：

```text
高质量 Semantic Occupancy Ground Truth
```

用于：

* Camera-only Occupancy
* BEV Occupancy
* Sparse4D / SparseDrive / BEVFormer
* End-to-End 自动驾驶
* 世界模型（World Model）
* Occupancy Prediction

最终输出：

```python
occ.shape = [X, Y, Z]
```

其中每个 voxel 包含：

```text
semantic occupancy label
```

例如：

```python
0 = unknown
1 = free
2 = road
3 = sidewalk
4 = building
5 = vehicle
6 = pedestrian
7 = vegetation
8 = pole
9 = traffic sign
```

---

# 2. 为什么不建议基于 LiDAR 生成 OCC

LiDAR 只适合：

```text
感知
```

不适合：

```text
生成 GT
```

原因：

* 遮挡区域不可见
* 远距离点云稀疏
* 只能观测表面
* unknown 区域过多
* 无法获得完整实体体积

例如：

```text
LiDAR 只能打到车前表面
车内部和车后方不可见
```

因此推荐：

```text
直接利用 CARLA GT Geometry
生成完整 Semantic Occupancy GT
```

---

# 3. 推荐最终方案

推荐：

```text
CARLA GT Geometry
        ↓
Offline Voxelization
        ↓
Semantic Occupancy GT
```

核心来源：

* waypoint → road
* actor bbox → dynamic object
* get_level_bbs → static object

并采用：

```text
采集阶段：
只保存 GT annotation

后处理阶段：
离线生成 OCC
```

---

# 4. 整体 Pipeline

完整推荐 Pipeline：

```text
CARLA Simulator
        ↓
获取 static map info
        ↓
生成 static_occ
        ↓
保存 static_occ.npy

---------------------------------

CARLA 采集阶段
        ↓
保存 ego pose
保存 dynamic actor annotation

---------------------------------

离线 OCC 后处理
        ↓
读取 static_occ
        ↓
根据 ego pose 做局部过滤
        ↓
填 dynamic actor voxel
        ↓
生成当前帧 occ.npy
```

---

# 5. OCC 类别设计

推荐：

```python
UNKNOWN_LABEL    = 0
FREE_LABEL       = 1

ROAD_LABEL       = 2
SIDEWALK_LABEL   = 3
BUILDING_LABEL   = 4
VEHICLE_LABEL    = 5
PEDESTRIAN_LABEL = 6
VEGETATION_LABEL = 7
POLE_LABEL       = 8
TRAFFIC_SIGN     = 9
```

---

# 6. OCC 空间配置

推荐：

```python
pc_range = [-50, -50, -5, 50, 50, 3]

voxel_size = [0.5, 0.5, 0.5]
```

得到：

```python
occ_shape = [
    200,
    200,
    16,
]
```

含义：

```text
X方向：200 voxel
Y方向：200 voxel
Z方向：16 voxel
```

---

# 7. 为什么必须离线生成 OCC

实时生成 OCC 会涉及：

* 大量 voxel filling
* road waypoint 遍历
* static object filling
* Python 三重循环
* 大量 world_to_voxel

会导致：

```text
CARLA FPS 下降
同步模式卡顿
采集速度变慢
```

因此：

```text
强烈推荐：
采集阶段只保存 annotation
离线后处理生成 OCC
```

---

# 8. 采集阶段保存什么

当前 OCC-only 方案：

```text
只保存 OCC 所需 GT
```

不需要：

* camera
* lidar
* imu
* gnss

除非后续训练需要。

---

# 9. 采集阶段保存内容

## 9.1 地图名称

例如：

```text
Town10HD
```

因为：

```text
static_occ 与地图强相关
```

---

## 9.2 Ego Pose

例如：

```json
{
  "ego_pose": {
    "location": [x, y, z],
    "rotation": [roll, pitch, yaw]
  }
}
```

原因：

```text
OCC 通常是 ego-centric
```

---

## 9.3 Dynamic Actor Annotation

保存：

* id
* category
* location
* rotation
* extent

例如：

```json
{
  "id": 123,
  "type": "vehicle",
  "location": [10.2, 3.5, 0.1],
  "rotation": [0, 0, 90],
  "extent": [2.1, 0.9, 0.8]
}
```

---

# 10. 推荐目录结构

```text
occ_dataset/

    map/
        Town10HD_static_bbs.pkl
        Town10HD_static_occ.npy

    frames/
        000001.json
        000002.json
        000003.json

    occ/
        000001.npy
        000002.npy
        000003.npy
```

---

# 11. OCC 初始化

```python
import numpy as np

occ = np.zeros(
    occ_shape,
    dtype=np.uint8,
)

occ[:] = UNKNOWN_LABEL
```

---

# 12. world_to_voxel

```python
def world_to_voxel(
    x,
    y,
    z,
    pc_range,
    voxel_size,
    occ_shape,
):

    ix = int((x - pc_range[0]) / voxel_size[0])
    iy = int((y - pc_range[1]) / voxel_size[1])
    iz = int((z - pc_range[2]) / voxel_size[2])

    if (
        ix < 0 or ix >= occ_shape[0]
        or iy < 0 or iy >= occ_shape[1]
        or iz < 0 or iz >= occ_shape[2]
    ):
        return None

    return ix, iy, iz
```

---

# 13. Road OCC（Waypoint）

# 13.1 Waypoint 是什么

Waypoint 是：

```text
CARLA 道路车道中心线点
```

包含：

* location
* rotation
* lane_width
* lane_type

---

# 13.2 获取 waypoint

```python
carla_map = world.get_map()

waypoints = carla_map.generate_waypoints(0.5)
```

---

# 13.3 为什么不能直接 road bbox filling

因为：

```text
road 是地面薄表面
不是实体立方体
```

如果直接 bbox filling：

```text
会把空气也填成 road
```

因此：

```text
只填地面附近 voxel
```

---

# 13.4 Road OCC 核心思想

```text
沿 waypoint 中心线
根据 lane_width 横向展开
只填地面薄层 voxel
```

---

# 13.5 Road OCC 示例代码

```python
import numpy as np
import carla


def fill_road_occ_from_waypoints(
    occ,
    world,
    pc_range,
    voxel_size,
    road_label=ROAD_LABEL,
    waypoint_gap=0.5,
    z_thickness=0.2,
):

    carla_map = world.get_map()

    waypoints = carla_map.generate_waypoints(
        waypoint_gap
    )

    for wp in waypoints:

        if wp.lane_type != carla.LaneType.Driving:
            continue

        loc = wp.transform.location

        yaw = np.deg2rad(
            wp.transform.rotation.yaw
        )

        lane_width = wp.lane_width

        right = np.array([
            np.sin(yaw),
            -np.cos(yaw),
        ])

        sample_step = min(
            voxel_size[0],
            voxel_size[1],
        ) * 0.5

        offsets = np.arange(
            -lane_width / 2.0,
            lane_width / 2.0 + sample_step,
            sample_step,
        )

        z_values = np.arange(
            loc.z - z_thickness,
            loc.z + z_thickness + voxel_size[2],
            voxel_size[2],
        )

        for offset in offsets:

            x = loc.x + right[0] * offset
            y = loc.y + right[1] * offset

            for z in z_values:

                idx = world_to_voxel(
                    x,
                    y,
                    z,
                    pc_range,
                    voxel_size,
                    occ.shape,
                )

                if idx is None:
                    continue

                ix, iy, iz = idx

                if occ[ix, iy, iz] in [
                    UNKNOWN_LABEL,
                    FREE_LABEL,
                ]:
                    occ[ix, iy, iz] = road_label
```

---

# 14. Dynamic Actor OCC

通过：

```python
world.get_actors()
```

获取：

* vehicle
* pedestrian

---

# 14.1 获取 Dynamic Actor

```python
actors = world.get_actors()

vehicles = actors.filter("*vehicle*")

walkers = actors.filter("*walker*")
```

---

# 14.2 Dynamic Actor Filling

```python
def fill_actor_bbox(
    occ,
    actor,
    label,
    pc_range,
    voxel_size,
):

    bbox = actor.bounding_box

    transform = actor.get_transform()

    verts = bbox.get_world_vertices(
        transform
    )

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    z_min = min(zs)
    z_max = max(zs)

    x_values = np.arange(
        x_min,
        x_max,
        voxel_size[0],
    )

    y_values = np.arange(
        y_min,
        y_max,
        voxel_size[1],
    )

    z_values = np.arange(
        z_min,
        z_max,
        voxel_size[2],
    )

    for x in x_values:
        for y in y_values:
            for z in z_values:

                idx = world_to_voxel(
                    x,
                    y,
                    z,
                    pc_range,
                    voxel_size,
                    occ.shape,
                )

                if idx is None:
                    continue

                ix, iy, iz = idx

                occ[ix, iy, iz] = label
```

---

# 15. Static Object OCC

通过：

```python
world.get_level_bbs()
```

获取：

* building
* static vehicle
* fence
* pole
* traffic sign

---

# 15.1 为什么不能直接处理全地图 static object

因为：

```text
Town10HD static object 数量非常大
```

如果：

```text
每帧对全地图 voxel filling
```

会非常慢。

因此：

```text
必须做 spatial filtering
```

---

# 15.2 推荐方案

```text
static_occ 只生成一次

每帧：
只过滤 ego 附近 static object
```

---

# 15.3 获取 Static Object

```python
bbs = world.get_level_bbs(
    carla.CityObjectLabel.Buildings
)
```

```python
bbs = world.get_level_bbs(
    carla.CityObjectLabel.Car
)
```

---

# 15.4 world_to_ego

```python
import numpy as np
import math


def yaw_to_rotmat(yaw_deg):
    yaw = math.radians(yaw_deg)

    c = math.cos(yaw)
    s = math.sin(yaw)

    return np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1],
    ], dtype=np.float32)


def world_to_ego_point(point, ego_transform):

    ego_loc = ego_transform.location
    ego_rot = ego_transform.rotation

    p = np.array([
        point.x - ego_loc.x,
        point.y - ego_loc.y,
        point.z - ego_loc.z,
    ], dtype=np.float32)

    R = yaw_to_rotmat(ego_rot.yaw)

    return R @ p
```

---

# 15.5 Static Object Spatial Filtering

```python
def filter_bbs_by_pc_range(
    bbs,
    ego_transform,
    pc_range,
    margin=5.0,
):

    valid_bbs = []

    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    for bbox in bbs:

        center_ego = world_to_ego_point(
            bbox.location,
            ego_transform,
        )

        x, y, z = center_ego

        if (
            x_min - margin <= x <= x_max + margin
            and y_min - margin <= y <= y_max + margin
            and z_min - margin <= z <= z_max + margin
        ):
            valid_bbs.append(bbox)

    return valid_bbs
```

---

# 15.6 为什么需要 margin

因为：

```text
有些大物体：
中心点在 OCC 外
但边缘进入 OCC 区域
```

因此：

```text
需要额外 margin
```

推荐：

```python
margin = 5m ~ 10m
```

---

# 15.7 Static Object Filling

```python
def fill_static_bbox(
    occ,
    bbox,
    label,
    pc_range,
    voxel_size,
):

    verts = bbox.get_world_vertices(
        carla.Transform()
    )

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    z_min = min(zs)
    z_max = max(zs)

    x_values = np.arange(
        x_min,
        x_max,
        voxel_size[0],
    )

    y_values = np.arange(
        y_min,
        y_max,
        voxel_size[1],
    )

    z_values = np.arange(
        z_min,
        z_max,
        voxel_size[2],
    )

    for x in x_values:
        for y in y_values:
            for z in z_values:

                idx = world_to_voxel(
                    x,
                    y,
                    z,
                    pc_range,
                    voxel_size,
                    occ.shape,
                )

                if idx is None:
                    continue

                ix, iy, iz = idx

                occ[ix, iy, iz] = label
```

---

# 16. 如何避免 OCC 生成过慢

推荐：

```text
static_occ 只生成一次
```

例如：

```python
static_occ = build_static_occ_once(world)

np.save(
    "Town10HD_static_occ.npy",
    static_occ,
)
```

---

# 17. 每帧 OCC 生成流程

每帧：

```python
occ = static_occ.copy()

fill_dynamic_actor_occ(
    occ,
    frame_actor_annotation,
)
```

因此：

```text
每帧只需要处理 dynamic actor
```

速度会非常快。

---

# 18. 推荐最终后处理 Pipeline

推荐：

```text
读取 Town10HD_static_occ.npy
        ↓
读取 frame json
        ↓
恢复 ego pose
        ↓
过滤 local static object
        ↓
填 dynamic actor
        ↓
生成当前帧 occ.npy
```

---

# 19. OCC 保存

推荐：

```python
np.save(
    "occ.npy",
    occ,
)
```

或者：

```python
import pickle

pickle.dump(
    occ,
    open("occ.pkl", "wb")
)
```

---

# 20. 当前方案优点

当前方案：

```text
完整
可扩展
高质量
工程可落地
```

主要优点：

* 不依赖 LiDAR
* OCC 离线生成
* static_occ 只生成一次
* 支持大规模数据集生成
* 支持 semantic occupancy
* 支持 camera-only occupancy
* 支持动态目标
* 支持静态地图
* 支持 ego-centric occupancy

---

# 21. 最终总结

当前推荐最终方案：

```text
CARLA GT Geometry
        ↓
Offline Voxelization
        ↓
Semantic Occupancy GT
```

推荐结构：

```text
static map:
    waypoint
    get_level_bbs

per frame:
    ego pose
    dynamic actor annotation

offline:
    local filtering
    voxel filling
    semantic occupancy
```

这是当前：

```text
最完整
最合理
最工程化
最适合大规模生成 OCC 数据集
```

的 CARLA OCC 方案。
