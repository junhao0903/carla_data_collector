# CARLA Data Collector

CARLA 0.9.14 仿真数据自动采集工具。

## 快速开始

```bash
bash scripts/start_carla.sh    # 启动 CARLA（后台运行）
python run.py                   # 运行采集（默认 5 秒）
bash scripts/stop_carla.sh     # 停止 CARLA
```

## 输出结构

```
output/<run_id>/
├── CAM_FRONT/
│   ├── original/         原始 RGB 图像 (.jpg)
│   ├── depth/        深度图 (.npy, 浮点米制)
│   ├── depth_viz/    深度可视化 (.png 灰度)
│   ├── semantic/     语义分割标签 (.png 灰度, tag=0-28)
│   ├── semantic_viz/ 语义分割可视化 (.png 彩色)
│   └── annotations/  2D/3D 目标检测真值 (.json)
├── LIDAR_TOP/
│   ├── original/         语义激光点云 (.npy, N×6: x,y,z,cos,tag,idx)
│   └── annotations/  3D 目标真值 (.json)
├── GNSS/data.csv     全球定位
├── IMU/data.csv      惯性测量
├── OCC_GT/           3D 语义占用栅格 (.npy)
├── OCC_GT_viz/       OCC 俯视可视化 (.png)
└── ego_trajectory.csv 自车轨迹
```

## 配置

主配置：`config/main/default.yaml`，传感器布局：`config/sensor/nuscenes.yaml`。

```bash
python run.py                          # 使用默认配置
python run.py config/main/other.yaml   # 使用其他主配置
```

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
