import os
import argparse
import numpy as np
from pyquaternion import Quaternion
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--frame_len', type=int, default=5, help='Number of frames to process at once')
    parser.add_argument('--data_root', type=str, default='./data/kitti', help='Path to the KITTI dataset root directory')
    parser.add_argument('--data_root_raw', type=str, default='./data/kitti/kitti_raw', help='Path to the raw KITTI dataset root directory')
    parser.add_argument('--save_path', type=str, default='./data/kitti/kitti_data_info.pkl', help='Path to save the generated data info')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mapping_dir = os.path.join(args.data_root, 'devkit_object', 'mapping')
    mapping_raw = []
    with open(os.path.join(mapping_dir, 'train_mapping.txt'), 'r') as f:
        for line in f:
            mapping_raw.append(line.strip().split())
    with open(os.path.join(mapping_dir, 'train_rand.txt'), 'r') as f:
        text = f.readline()
        mapping_id = [int(idx)-1 for idx in text.strip().split(',')]

    data_info = {'kitti_object': [], 'kitti_raw': {}}
    origin = None
    for i, idx in enumerate(mapping_id):
        raw_dir = os.path.join(args.data_root_raw, mapping_raw[idx][0], mapping_raw[idx][1])
        scene = mapping_raw[idx][1]
        if os.path.exists(raw_dir):
            if data_info['kitti_raw'].get(scene, None) is None:
                data_info['kitti_raw'][scene] = {}

            sample_idx = int(mapping_raw[idx][2])
            max_sample_idx = len(os.listdir(os.path.join(raw_dir, 'velodyne_points', 'data')))
            start = max(0, sample_idx - (args.frame_len - 1) // 2)
            end = min(max_sample_idx - 1, sample_idx + (args.frame_len - 1) // 2)
            for j in range(start, end + 1):
                name = str(j).zfill(10)
                if data_info['kitti_raw'][scene].get(name, None) is None:
                    with open(os.path.join(raw_dir, 'velodyne_points', 'timestamps.txt'), 'r') as f:
                        text = f.readlines()
                        timestamp = text[j].strip().split()[1].split(':')
                        timestamp = float(timestamp[1]) * 60 + float(timestamp[2])

                    with open(os.path.join(raw_dir, 'oxts', 'data', name + '.txt'), 'r') as f:
                        oxt = f.readlines()[0].strip().split()
                        lat, lon, alt, roll, pitch, yaw = [float(info) for info in oxt[:6]]
                        scale = np.cos(lat * np.pi / 180.)
                        er = 6378137.  # earth radius (approx.) in meters
                        tx = scale * lon * np.pi * er / 180.
                        ty = scale * er * np.log(np.tan((90. + lat) * np.pi / 360.))
                        tz = alt
                        t = np.array([tx, ty, tz])
                        if origin is None:
                            origin = t
                        t = t - origin
                        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
                        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
                        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                        R = Rz @ Ry @ Rx
                        ego2global = np.eye(4)
                        ego2global[:3, :3] = R
                        ego2global[:3, 3] = t
                        
                    data_info['kitti_raw'][scene][name] = {
                        'timestamp': timestamp,
                        'lidar_path': os.path.join(raw_dir, 'velodyne_points', 'data', name + '.bin'),
                        'img_path': os.path.join(raw_dir, 'image_02', 'data', name + '.png'),
                        'ego2global': ego2global
                    }

            with open(os.path.join(args.data_root, 'training', 'calib', str(i).zfill(6) + '.txt')) as f:
                lines = f.readlines()
                cam2img = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
                R0_rect = np.eye(4)
                R0_rect[:3, :3] = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
                lidar2cam = np.eye(4)
                lidar2cam[:3, :] = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
                lidar2cam = R0_rect @ lidar2cam
                ego2lidar = np.eye(4)
                ego2lidar[:3, :] = np.array([float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])

            data_info['kitti_object'].append({'scene': scene, 'sample_idx': sample_idx, 'start_idx': start, 'end_idx': end, 'name': str(i).zfill(6),
                                              'cam2img': cam2img, 'lidar2cam': lidar2cam, 'ego2lidar': ego2lidar,
                                              'label': os.path.join(args.data_root, 'training', 'label_2', str(i).zfill(6) + '.txt')})
            
    with open(args.save_path, 'wb') as f:
        pickle.dump(data_info, f)
    print(f"Data info saved to {args.save_path}.")