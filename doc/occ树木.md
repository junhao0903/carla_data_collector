# CARLA OCC 中 Vegetation 的处理方案

# 1. 为什么 Vegetation 不能直接 bbox filling

对于：

* building
* vehicle
* pedestrian
* pole

通常：

```text
bbox filling
```

效果已经不错。

但是：

```text
Vegetation（树木、灌木）
```

不能直接：

```text
整个 bbox 全填充
```

原因：

```text
Vegetation 的真实几何：

树干占据空间很小
树冠占据空间很大
bbox 内大量区域实际上是空气
```

因此：

```text
直接 bbox filling
会产生大量错误 occupancy
```

---

# 2. 直接 bbox filling 的问题

例如：

```text
树木 bbox：

############
############
############
############
############
############
```

如果：

```text
整个 bbox 都填 vegetation
```

会导致：

* 树干异常粗
* 树冠变成立方体
* 树下空间错误占据
* BEV 中 vegetation 非常奇怪
* OCC 质量明显下降

---

# 3. 推荐最终方案

推荐：

```text
bbox height heuristic
```

即：

```text
bbox 下部：
只填中心细柱

bbox 上部：
填树冠区域

地面层：
保留 road / sidewalk
```

这是：

```text
最稳定
最工程化
最简单
```

的 vegetation OCC 方案。

---

# 4. 核心思想

Vegetation 不再：

```text
整体 bbox filling
```

而是：

```text
下部：tree trunk
上部：tree crown
```

示意：

```text
        tree crown
      ################
    ####################
           ||||
           ||||
           ||||  trunk
           ||||
```

---

# 5. 为什么这种方案有效

因为 CARLA 中大部分 vegetation asset：

```text
树干在下
树冠在上
```

因此：

```text
bbox 的统计规律非常稳定
```

所以：

```text
可以基于 bbox 高度比例
近似 trunk 和 crown
```

---

# 6. 推荐 Geometry Heuristic

假设：

```python
bbox:
    width_x
    width_y
    height
```

定义：

---

## 6.1 Tree Trunk

树干：

```text
bbox 中心
半径很小
位于下部
```

推荐：

```python
trunk_radius = min(width_x, width_y) * 0.08
```

通常：

```text
5% ~ 10%
```

效果最好。

---

## 6.2 Trunk Height

推荐：

```python
trunk_z_max = z_min + H * 0.45
```

即：

```text
bbox 下 45%
属于 trunk
```

---

## 6.3 Crown Region

推荐：

```python
crown_z_min = z_min + H * 0.35
```

即：

```text
bbox 上半部分
属于 crown
```

---

# 7. Ground Layer 不应该被 Vegetation 覆盖

非常重要：

```text
树冠正下方不应该为空
```

也不应该：

```text
被 vegetation 覆盖
```

而应该：

```text
保留 road / sidewalk / terrain
```

因此：

```text
vegetation filling
不能覆盖 road
```

推荐：

```python
if occ[ix, iy, iz] in [
    UNKNOWN_LABEL,
    FREE_LABEL,
]:
    occ[ix, iy, iz] = VEGETATION_LABEL
```

不要：

```python
occ[ix, iy, iz] = VEGETATION_LABEL
```

---

# 8. 推荐 Vegetation OCC Filling 逻辑

推荐：

```text
bbox 下部：
只填中心细柱

bbox 上部：
允许 vegetation filling

地面层：
保留 road / sidewalk
```

最终效果：

```text
z低层：road
z中层：细 trunk
z高层：tree crown
```

---

# 9. 为什么不是真正识别 trunk 和 crown

当前方案：

```text
并不是真正 mesh segmentation
```

而是：

```text
procedural geometry heuristic
```

即：

```text
基于 bbox 统计规律
近似 trunk 和 crown
```

原因：

```text
CARLA bbox 本身：
只有：

中心
长宽高
朝向

并不包含 mesh 几何
```

因此：

```text
只能做 heuristic approximation
```

---

# 10. 推荐最终方案（方案2）

推荐：

```text
bbox 下部：
中心细柱

bbox 上部：
较宽 vegetation filling
```

而不是：

```text
完整 bbox filling
```

这是：

```text
最简单
最稳定
最适合工程落地
```

的 vegetation OCC 方案。

---

# 11. Vegetation OCC 示例代码

```python
import numpy as np


def fill_vegetation_occ(
    occ,
    bbox,
    pc_range,
    voxel_size,
    vegetation_label,
):

    verts = bbox.get_world_vertices(
        carla.Transform()
    )

    xs = np.array([v.x for v in verts])
    ys = np.array([v.y for v in verts])
    zs = np.array([v.z for v in verts])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()

    width_x = x_max - x_min
    width_y = y_max - y_min
    H = z_max - z_min

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5

    trunk_radius = min(width_x, width_y) * 0.08

    trunk_z_max = z_min + H * 0.45

    crown_z_min = z_min + H * 0.35

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

                dx = x - cx
                dy = y - cy

                # trunk region
                in_trunk = (
                    dx * dx + dy * dy
                    <= trunk_radius * trunk_radius
                    and z <= trunk_z_max
                )

                # crown region
                in_crown = (
                    z >= crown_z_min
                )

                if not (in_trunk or in_crown):
                    continue

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

                # 不覆盖 road
                if occ[ix, iy, iz] in [
                    UNKNOWN_LABEL,
                    FREE_LABEL,
                ]:
                    occ[ix, iy, iz] = vegetation_label
```

---

# 12. 推荐参数

推荐：

```python
trunk_radius_ratio = 0.05 ~ 0.1

trunk_height_ratio = 0.4 ~ 0.5

crown_start_ratio = 0.3 ~ 0.4
```

推荐默认：

```python
trunk_radius_ratio = 0.08
trunk_height_ratio = 0.45
crown_start_ratio = 0.35
```

---

# 13. 当前方案优点

当前方案：

```text
实现简单
速度快
不依赖 mesh
不依赖 semantic lidar
工程可落地
```

同时：

```text
相比完整 bbox filling
vegetation occupancy 质量会明显提升
```

---

# 14. 当前方案缺点

当前方案：

```text
仍然不是真实 vegetation geometry
```

因为：

```text
没有真实 mesh
```

因此：

```text
tree crown 仍然是 heuristic approximation
```

但：

```text
相比 bbox filling
已经会好很多
```

---

# 15. 更高级方案（未来可扩展）

未来可以继续扩展：

---

## 15.1 Mesh Voxelization

直接：

```text
voxelize vegetation mesh
```

最准。

但：

```text
实现复杂度很高
```

---

## 15.2 Semantic LiDAR Refinement

例如：

```text
bbox 提供 vegetation prior
semantic lidar 修正边界
```

---

## 15.3 Asset-specific Geometry

例如：

```text
oak
pine
bush
```

分别不同 geometry。

---

# 16. 最终总结

当前推荐最终 vegetation OCC 方案：

```text
bbox 下部：
中心细柱

bbox 上部：
vegetation filling

地面层：
保留 road / sidewalk
```

而不是：

```text
完整 bbox filling
```

这是当前：

```text
最简单
最稳定
最工程化
```

的 CARLA Vegetation OCC 方案。

