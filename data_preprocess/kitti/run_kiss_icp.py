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

    data_root = './data/kitti/kitti_raw'
    dates = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]

    for date in dates:
        drives_dirs = os.listdir(os.path.join(data_root, date))
        drives_dirs = [d for d in drives_dirs if os.path.isdir(os.path.join(data_root, date, d))]

        for num, drive_dir in enumerate(drives_dirs):

            save_path = os.path.join(data_root, date, drive_dir, 'poses.txt')
            if os.path.exists(save_path):
                print(f'skipped {date} {drive_dir} ({num+1}/{len(drives_dirs)})')
                continue

            ts_path = os.path.join(data_root, date, drive_dir, 'velodyne_points', 'timestamps.txt')
            with open(ts_path, 'r') as f:
                text = f.readlines()
            timestamps = []
            for t in text:
                if len(t.strip().split()) > 1:
                    ts = t.strip().split()[1].split(':')
                    ts = float(ts[1]) * 60 + float(ts[2])
                    timestamps.append(ts)
                else:
                    timestamps.append(None)
            if len(text) != len(timestamps):
                print(f'warning: timestamps length mismatch {date} {drive_dir} ({num+1}/{len(drives_dirs)})')
                continue

            lidar_dir = os.path.join(data_root, date, drive_dir, 'velodyne_points', 'data')
            kiss_config = KISSConfig()
            kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
            odometry = KissICP(config=kiss_config)

            poses = []
            for i in range(len(timestamps)):
                if timestamps[i] is None:
                    poses.append(np.eye(4))
                else:
                    lidar_path = os.path.join(lidar_dir, str(i).zfill(10) + '.bin')
                    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
                    x = points[:, 0]
                    y = points[:, 1]
                    yaw = -np.arctan2(y, x)
                    points_ts = 0.5 * (yaw / np.pi + 1.0)
                    odometry.register_frame(correct_kitti_scan(np.copy(points).astype(np.float64)), timestamps=points_ts)
                    pose = odometry.last_pose
                    poses.append(pose)

            np.savetxt(save_path, np.asarray(poses).reshape(len(poses), 16))
            print(f'processed {date} {drive_dir} ({num+1}/{len(drives_dirs)})')