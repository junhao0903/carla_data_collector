#!/usr/bin/env python3
"""Convert OCC npy frames to ROS PointCloud2 and save as rosbag.

Usage:
    python tools/occ_to_rosbag.py output/<run> [--topic /occ/points] [--fps 20]
"""
import argparse
import json
import os
import struct
import sys
import time
from collections import namedtuple

import numpy as np

import rosbag
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# OCC label → readable name
LABEL_NAMES = {
    0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building",
    4: "wall", 5: "fence", 6: "pole", 7: "traffic_light",
    8: "traffic_sign", 9: "vegetation", 10: "terrain", 11: "sky",
    12: "pedestrian", 13: "rider", 14: "car", 15: "truck",
    16: "bus", 17: "train", 18: "motorcycle", 19: "bicycle",
    20: "static", 21: "dynamic", 22: "other",
    23: "water", 24: "road_line", 25: "ground",
    26: "bridge", 27: "rail_track", 28: "guard_rail",
}


def build_pointcloud(points, labels, frame_id="ego", stamp=None):
    """Build a sensor_msgs/PointCloud2 message with x,y,z,label fields.

    Args:
        points: (N, 3) float32 array in ego frame (X=forward, Y=left, Z=up)
        labels: (N,) uint8 array of semantic tags
        frame_id: TF frame name
        stamp: rospy.Time or None

    Returns:
        PointCloud2 message
    """
    # Fields: x, y, z (float32), label (uint8), padding (3 bytes for alignment)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="label", offset=12, datatype=PointField.UINT8, count=1),
    ]
    # 4 float32 fields + 1 uint8 = 13 bytes. Round up to 16 for alignment
    point_step = 16

    n = len(points)
    data = bytearray(n * point_step)
    for i in range(n):
        offset = i * point_step
        struct.pack_into("<fffB", data, offset,
                         float(points[i, 0]),
                         float(points[i, 1]),
                         float(points[i, 2]),
                         int(labels[i]))
        # bytes 13-15 are padding (already zero)

    header = Header(frame_id=frame_id, stamp=stamp if stamp else rospy.Time.from_sec(time.time()))
    return PointCloud2(
        header=header,
        height=1,
        width=n,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * n,
        data=bytes(data),
        is_dense=True,
    )


def run(run_dir, topic="/occ/points", fps=20, sample=1):
    occ_dir = os.path.join(run_dir, "OCC", "original")
    if not os.path.isdir(occ_dir):
        print(f"No OCC npy files in {occ_dir}")
        return

    npy_files = sorted(f for f in os.listdir(occ_dir) if f.endswith(".npy"))
    if not npy_files:
        print("No .npy files found")
        return

    # Load OCC metadata
    meta_path = os.path.join(run_dir, "OCC", "occ_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        pc_range = meta.get("pc_range", [-50, -50, -5, 50, 50, 3])
        voxel_size = meta.get("voxel_size", [0.5, 0.5, 0.5])
    else:
        pc_range = [-50, -50, -5, 50, 50, 3]
        voxel_size = [0.5, 0.5, 0.5]

    bag_path = os.path.join(run_dir, "OCC", "occ.bag")
    os.makedirs(os.path.dirname(bag_path), exist_ok=True)
    print(f"Converting {len(npy_files)} OCC frames → {bag_path}")
    print(f"  OCC range: {pc_range}, voxel: {voxel_size}, sample: {sample}")
    print(f"  Topic: {topic}")

    dt = rospy.Duration.from_sec(1.0 / fps)
    start_stamp = rospy.Time.from_sec(time.time()) - dt * len(npy_files)

    from tqdm import tqdm

    bag = rosbag.Bag(bag_path, "w")
    try:
        for idx, fname in enumerate(tqdm(npy_files, desc="Writing rosbag")):
            occ = np.load(os.path.join(occ_dir, fname))

            # Extract occupied voxels with downsampling
            xs, ys, zs_n = np.where(occ > 0)
            mask = (xs % sample == 0) & (ys % sample == 0) & (zs_n % sample == 0)
            xs, ys, zs_n = xs[mask], ys[mask], zs_n[mask]
            labels = occ[xs, ys, zs_n]

            if len(xs) == 0:
                continue

            # Voxel index → ego frame position
            px = pc_range[0] + (xs + 0.5) * voxel_size[0]  # forward
            py = pc_range[1] + (ys + 0.5) * voxel_size[1]  # left
            pz = pc_range[2] + (zs_n + 0.5) * voxel_size[2]  # up
            points = np.stack([px, py, pz], axis=1).astype(np.float32)

            stamp = start_stamp + dt * idx
            cloud = build_pointcloud(points, labels, frame_id="ego", stamp=stamp)
            bag.write(topic, cloud, stamp)

        print(f"Done: {len(npy_files)} frames → {bag_path}")
    finally:
        bag.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert OCC npy to ROS PointCloud2 rosbag")
    parser.add_argument("run_dir", help="Path to collection run directory")
    parser.add_argument("--topic", default="/occ/points",
                        help="ROS topic name (default: /occ/points)")
    parser.add_argument("--fps", type=float, default=20,
                        help="Frame rate for timestamps (default: 20)")
    parser.add_argument("--sample", type=int, default=1,
                        help="Voxel downsampling step (default: 1)")
    args = parser.parse_args()

    run(args.run_dir, args.topic, args.fps, args.sample)


if __name__ == "__main__":
    main()
