#!/usr/bin/env python3
import os
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP


def correct_kitti_scan(frame: np.ndarray):
    from kiss_icp.pybind import kiss_icp_pybind
    assert frame.dtype == np.float64
    return np.asarray(kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame)))


if __name__ == "__main__":

    # dates = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]
    data_root = './data/kitti/kitti_raw'
    dates = ["2011_09_26"]

    for date in dates:
        drives_dirs = os.listdir(os.path.join(data_root, date))

        for drive_dir in drives_dirs:
            lidar_dir = os.path.join(data_root, date, drive_dir, 'velodyne_points', 'data')
            ts_path = os.path.join(data_root, date, drive_dir, 'velodyne_points', 'timestamps.txt')
            kiss_config = KISSConfig()
            kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
            odometry = KissICP(config=kiss_config)

            with open(ts_path, 'r') as f:
                text = f.readlines()
            timestamps = [t.strip().split()[1].split(':') for t in text]
            timestamps = [float(t[1]) * 60 + float(t[2]) for t in timestamps]

            poses = []
            for i in range(len(timestamps)):
                lidar_path = os.path.join(lidar_dir, str(i).zfill(10) + '.bin')
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                x = points[:, 0]
                y = points[:, 1]
                yaw = -np.arctan2(y, x)
                points_ts = 0.5 * (yaw / np.pi + 1.0)
                odometry.register_frame(correct_kitti_scan(np.copy(points).astype(np.float64)), timestamps=points_ts)
                pose = odometry.last_pose
                poses.append(pose)

            save_path = os.path.join(data_root, date, drive_dir, 'poses.txt')
            np.savetxt(save_path, np.asarray(poses).reshape(len(poses), 16))