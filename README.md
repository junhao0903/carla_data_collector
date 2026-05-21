# CARLA Data Collector

CARLA 0.9.x 仿真数据自动采集工具。

## 快速开始

> **使用前需配置 CARLA 路径**：本项目依赖 CARLA 0.9.x，请将以下两个文件中的路径改为你的实际路径：
>
> - `scripts/start_carla.sh` → `CARLA_SCRIPT=/你的路径/carlaUE4.sh`
> - `config/main/default.yaml` → `launch_command: ["/你的路径/carlaUE4.sh"]`（以及 `config/main/` 下其他 yaml）
```bash
conda activate carla
bash scripts/start_carla.sh        # 启动 CARLA（后台运行）
python run.py                       # 运行采集
bash scripts/stop_carla.sh         # 停止 CARLA

# 可使用其他主配置
python run.py config/main/other.yaml
```

## 后处理可视化

采完数据后，可单独运行后处理：

```bash
# 可视化（BEV 图、深度/语义着色、标注框，遵循 vis 开关）
python tools/npy2jpg.py output/指定文件夹
python tools/npy2jpg.py output/指定文件夹 --all   # 强制全开

# OCC 投影到相机图像
python tools/occ_projection.py output/指定文件夹
python tools/occ_projection.py output/指定文件夹 --channel CAM_FRONT_LEFT

# OCC 导出 ROS PointCloud2 rosbag
python tools/occ_to_rosbag.py output/指定文件夹
```

## 配置结构

```
config/
├── main/
│   └── default.yaml     # 主配置: CARLA连接、车辆、采集参数、ground_truth
├── sensor/
│   └── nuscenes.yaml    # 传感器布局: 每个传感器独立配置
└── filter/
    └── default.yaml     # 过滤 LiDAR 配置: annotation 过滤专用
```

传感器布局中每个相机可独立控制：

```yaml
sensors:
  - id: cam_front
    channel: CAM_FRONT
    modality: camera_rgb
    enabled: true
    depth: true          # 是否采集深度图
    depth_vis: true      # 是否生成深度可视化
    semantic: true       # 是否采集语义分割
    semantic_vis: true   # 是否生成语义可视化
    annotation_vis: true # 是否生成标注可视化
    transform: {...}
    output: {...}
```

顶层 `trajectory_vis: true` 控制轨迹可视化。

## 输出结构

```
output/<YYYYMMDD_HHMMSS>/
├── sensor_layout.yaml        # 传感器布局副本 (供后处理使用)
├── CAM_FRONT/
│   ├── original/             # RGB 图像 (.npy → .jpg)
│   ├── depth/                # 深度图 (.npy, 米制浮点)
│   ├── depth_viz/            # 深度可视化 (.png 灰度)
│   ├── semantic/             # 语义分割标签 (.png, tag 0-28)
│   ├── semantic_viz/         # 语义可视化 (.png 彩色)
│   ├── annotations/          # 2D/3D 标注 (.json)
│   └── annotations_viz/      # 标注可视化 (.jpg, 2D 框绘于原图)
├── LIDAR_TOP/                # 数据 LiDAR (传感器布局中配置)
│   ├── original/             # 点云 (.npy, N×4 或 N×6)
│   ├── annotations/          # 3D 标注 (.json, ego 坐标系)
│   └── annotations_viz/      # BEV 标注可视化 (.png)
├── LIDAR_FILTER/             # 过滤 LiDAR (output: true 时出现)
│   ├── original/             # 点云 (.npy)
│   ├── annotations/          # 3D 标注 (.json)
│   └── annotations_viz/      # BEV 可视化 (.png)
├── TRAJ/
│   ├── ego_trajectory.csv    # 自车轨迹
│   └── trajectory_viz/       # 轨迹 BEV 可视化 (.png)
├── GNSS/data.csv             # 全球定位 (lat/lon/alt)
├── IMU/data.csv              # 惯性测量 (加速度/角速度/罗盘)
├── OCC/
│   ├── annotations/       # 自车坐标系 actor 标注 (.json)
│   ├── original/          # 3D 占用栅格 (.npy, 200×200×30)
│   ├── occ_viz/           # OCC BEV 可视化 (.png)
│   ├── projection_CAM_FRONT/  # 投影到相机图像 (.jpg)
│   ├── occ.bag            # ROS PointCloud2 rosbag
│   └── occ_metadata.json  # OCC 参数
├── map/                       # 地图静态 OCC（自动生成）
│   └── Town10HD_static_occ.npy
```

## 坐标系

统一使用 **前-X / 左-Y / 上-Z** 左手坐标系。

### 各数据坐标系速查

| 数据 | X | Y | Z | roll(+) | pitch(+) | yaw(+) | 原点 |
|------|---|---|---|---------|---------|--------|------|
| LiDAR 点云 | 前 | 左 | 上 | — | — | — | LiDAR传感器 |
| LiDAR 标注 | 前 | 左 | 上 | 左翼下沉 | 抬头 | 机头左转 | 自车 |
| CAM 标注 (world) | 前 | 右(CARLA) | 上 | 右翼下沉(CARLA) | 抬头 | 机头右转(CARLA) | CARLA世界 |
| CAM 标注 (camera) | 前 | 左 | 上 | 左翼下沉(相对) | 抬头(相对) | 机头左转(相对) | 相机 |
| OCC 标注 | 前 | 左 | 上 | 左翼下沉 | 抬头 | 机头左转 | 自车 |
| ego_trajectory | 前 | 左 | 上 | 左翼下沉 | 抬头 | 机头左转 | CARLA世界 |
| IMU accel | 前 | 左 | 上 | — | — | — | 车体 |
| IMU gyro | 前(左滚+) | 左(抬头+) | 上(左转+) | — | — | — | 车体 |
| GNSS | — | — | — | — | — | — | WGS84 |
| OCC 栅格 | 前 | 左 | 上 | — | — | — | 自车 |

### CARLA 世界 → 本系统变换

CARLA (Unreal) 原始坐标系: X=前, Y=**右**, Z=上, roll=右翼下沉, yaw=机头右转 (右手系)

变换规则: **Y 取负, roll 取负, yaw 取负, pitch 不变**

### 标注 location 说明

- **动态 actor**: `location.z` = 几何中心 (pivot地面 + half_height)
- **静态 BBs**: `location.z` = 几何中心 (`bb.location.z`)
- **相机帧 Z**: `loc_world.z - cam.z + bbox_height/2` = 几何中心相对相机
- **OCC 标注 extent**: 半尺寸 (`bbox.extent`)，`x/y/z` = 半长/半宽/半高

### bbox_3d 格式

`{"x": full_length, "y": full_width, "z": full_height}` — 完整尺寸 (extent × 2)，仅相机/LiDAR 标注使用

### bbox_2d 格式 (仅相机标注)

`[x_min, y_min, x_max, y_max]` — 图像像素坐标, 左上角为原点

## 标注过滤

通过专用过滤 LiDAR (`config/filter/default.yaml`) 对 bbox 进行点云计数过滤:

- bbox 内 LiDAR 点数 < `min_points` (默认 10) → 丢弃
- 仅检查 XY 平面 (不计 Z)
- 过滤 LiDAR 独立于传感器布局, 不消耗数据 LiDAR 资源
- `output: true` 时同时输出该 LiDAR 的点云和标注

## 语义标签颜色对照

| Tag | 类别 | 颜色 | 色块 |
|-----|------|------|------|
| 0 | unlabeled | (0, 0, 0) | ![#000000](https://placehold.co/15x15/000000/000000.png) |
| 1 | road | (128, 64, 128) | ![#804080](https://placehold.co/15x15/804080/804080.png) |
| 2 | sidewalk | (244, 35, 232) | ![#F423E8](https://placehold.co/15x15/F423E8/F423E8.png) |
| 3 | building | (70, 70, 70) | ![#464646](https://placehold.co/15x15/464646/464646.png) |
| 4 | wall | (102, 102, 156) | ![#66669C](https://placehold.co/15x15/66669C/66669C.png) |
| 5 | fence | (190, 153, 153) | ![#BE9999](https://placehold.co/15x15/BE9999/BE9999.png) |
| 6 | pole | (153, 153, 153) | ![#999999](https://placehold.co/15x15/999999/999999.png) |
| 7 | traffic_light | (250, 170, 30) | ![#FAAA1E](https://placehold.co/15x15/FAAA1E/FAAA1E.png) |
| 8 | traffic_sign | (220, 220, 0) | ![#DCDC00](https://placehold.co/15x15/DCDC00/DCDC00.png) |
| 9 | vegetation | (107, 142, 35) | ![#6B8E23](https://placehold.co/15x15/6B8E23/6B8E23.png) |
| 10 | terrain | (152, 251, 152) | ![#98FB98](https://placehold.co/15x15/98FB98/98FB98.png) |
| 11 | sky | (70, 130, 180) | ![#4682B4](https://placehold.co/15x15/4682B4/4682B4.png) |
| 12 | pedestrian | (220, 20, 60) | ![#DC143C](https://placehold.co/15x15/DC143C/DC143C.png) |
| 13 | rider | (255, 0, 0) | ![#FF0000](https://placehold.co/15x15/FF0000/FF0000.png) |
| 14 | car | (0, 0, 142) | ![#00008E](https://placehold.co/15x15/00008E/00008E.png) |
| 15 | truck | (0, 0, 70) | ![#000046](https://placehold.co/15x15/000046/000046.png) |
| 16 | bus | (0, 60, 100) | ![#003C64](https://placehold.co/15x15/003C64/003C64.png) |
| 17 | train | (0, 80, 100) | ![#005064](https://placehold.co/15x15/005064/005064.png) |
| 18 | motorcycle | (0, 0, 230) | ![#0000E6](https://placehold.co/15x15/0000E6/0000E6.png) |
| 19 | bicycle | (119, 11, 32) | ![#770B20](https://placehold.co/15x15/770B20/770B20.png) |
| 20 | static | (110, 190, 160) | ![#6EBEA0](https://placehold.co/15x15/6EBEA0/6EBEA0.png) |
| 21 | dynamic | (170, 120, 50) | ![#AA7832](https://placehold.co/15x15/AA7832/AA7832.png) |
| 22 | other | (55, 90, 80) | ![#375A50](https://placehold.co/15x15/375A50/375A50.png) |
| 23 | water | (45, 60, 150) | ![#2D3C96](https://placehold.co/15x15/2D3C96/2D3C96.png) |
| 24 | road_line | (157, 234, 50) | ![#9DEA32](https://placehold.co/15x15/9DEA32/9DEA32.png) |
| 25 | ground | (81, 0, 81) | ![#510051](https://placehold.co/15x15/510051/510051.png) |
| 26 | bridge | (150, 100, 100) | ![#966464](https://placehold.co/15x15/966464/966464.png) |
| 27 | rail_track | (230, 150, 140) | ![#E6968C](https://placehold.co/15x15/E6968C/E6968C.png) |
| 28 | guard_rail | (180, 165, 180) | ![#B4A5B4](https://placehold.co/15x15/B4A5B4/B4A5B4.png) |

## OCC (3D Semantic Occupancy)

基于 CARLA GT Geometry 生成的 3D 语义占用栅格，用于 BEV/Occupancy 预测任务训练。

### 架构

```
CARLA GT Geometry
        ↓
waypoint → road / sidewalk
get_level_bbs → building, wall, fence, pole, traffic_sign, vegetation
        ↓
static_occ.npy (一次性预生成，全地图)
        ↓
采集阶段: 每帧保存 ego pose + dynamic/static actor 标注
        ↓
后处理: static_occ 裁剪 + 动态 actor 叠加 → 每帧 occ.npy
```