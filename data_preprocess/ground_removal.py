import os
import argparse
import pickle
import numpy as np
import pypatchworkpp


def ground_removal(lidar_with_sweeps, sample_idx, scene_save_dir):
    points_c = []
    num_points = []
    start = max(0, sample_idx - (args.frame_len - 1) // 2)
    end = min(len(lidar_with_sweeps), sample_idx + (args.frame_len + 1) // 2)
    for i in range(start, end):
        lidar_path = lidar_with_sweeps[i]['lidar_path']
        lidar2global = lidar_with_sweeps[i]['lidar2global']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        R = lidar2global[:3, :3]
        T = lidar2global[:3, 3]
        points[:, :3] = (R @ points[:, :3].T).T + T
        points_c.append(points)
        num_points.append(points.shape[0])

    points_c = np.concatenate(points_c, axis=0, dtype=np.float32)
    lidar2global = lidar_with_sweeps[sample_idx]['lidar2global']
    R_inv = lidar2global[:3, :3].T
    T_inv = -R_inv @ lidar2global[:3, 3]
    points_c[:, :3] = ((R_inv @ points_c[:, :3].T).T + T_inv).astype(np.float32)
    points_c = np.concatenate([points_c, np.arange(0, len(points_c))[:, None]], axis=1)

    params = pypatchworkpp.Parameters()
    params.verbose = False
    params.enable_RNR = False
    params.sensor_height = 1.723
    params.min_range = 1.0
    params.max_range = 64
    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
    PatchworkPLUSPLUS.estimateGround(points_c)
    # ground = PatchworkPLUSPLUS.getGround()
    # nonground = PatchworkPLUSPLUS.getNonground()
    ground_idx = np.array(PatchworkPLUSPLUS.getGroundIndices())
    # nonground_idx = PatchworkPLUSPLUS.getNongroundIndices()

    idx = 0
    for i in range(start, end):
        num = num_points[i - start]
        is_ground = np.zeros(num, dtype=bool)
        _ground_idx = ground_idx[(ground_idx >= idx) & (ground_idx < idx + num)] - idx
        is_ground[_ground_idx] = True
        idx += num
        lidar_path = lidar_with_sweeps[i]['lidar_path']
        save_path = os.path.join(scene_save_dir, os.path.basename(lidar_path))
        is_ground.tofile(save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--frame_len', type=int, default=11, help='Number of frames to process at once')
    parser.add_argument('--info_path', type=str, default='./data/nuscenes/nuscenes_data_info.pkl', help='Path to the NuScenes data info file')
    parser.add_argument('--save_dir', type=str, default='./data/nuscenes/is_ground', help='Directory to save the ground removal results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)
    
    for scene in data_info:
        scene_name = scene['scene_name']
        scene_save_dir = f"{args.save_dir}/{scene_name}"
        os.makedirs(scene_save_dir, exist_ok=True)

        lidar_with_sweeps = []
        sample_idx_list = []
        count = 0
        for sample in scene['samples']:
            for sweep in sample['lidar_sweep']:
                lidar_with_sweeps.append({
                    'timestamp': sweep['timestamp'],
                    'lidar_path': sweep['lidar_path'],
                    'lidar2global': sweep['ego2global'] @ sample['lidar2ego'],
                })
                count += 1
            lidar_with_sweeps.append({
                'timestamp': sample['timestamp'],
                'lidar_path': sample['lidar_path'],
                'lidar2global': sample['ego2global'] @ sample['lidar2ego'],
            })
            sample_idx_list.append(count)
            count += 1

        for sample_idx in sample_idx_list:
            ground_removal(lidar_with_sweeps, sample_idx, scene_save_dir)

        print(f"Processing scene: {scene_name}, total frames: {len(lidar_with_sweeps)}")

    print("Ground removal processing completed.")