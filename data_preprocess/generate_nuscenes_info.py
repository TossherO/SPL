import argparse
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--version', type=str, default='v1.0-trainval-select', help='NuScenes dataset version')
    parser.add_argument('--data_root', type=str, default='./data/nuscenes', help='Path to the NuScenes dataset root directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--save_path', type=str, default='./data/nuscenes/nuscenes_data_info.pkl', help='Path to save the generated data info')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=args.verbose)

    data_info = []
    for scene in nusc.scene:
        scene_samples = []
        sample_token = scene['first_sample_token']
        while sample_token != '':
            sample = nusc.get('sample', sample_token)

            timestamp = sample['timestamp'] / 1e6
            lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_path = nusc.get_sample_data_path(lidar['token'])
            
            lidar_calib = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = Quaternion(lidar_calib['rotation']).rotation_matrix
            lidar2ego[:3, 3] = lidar_calib['translation']

            pose = nusc.get('ego_pose', lidar['ego_pose_token'])
            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(pose['rotation']).rotation_matrix
            ego2global[:3, 3] = pose['translation']

            cams = {}
            for cam_name in sample['data']:
                if cam_name.startswith('CAM'):
                    cam = nusc.get('sample_data', sample['data'][cam_name])
                    img_path = nusc.get_sample_data_path(cam['token'])
                    calib = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
                    cam_pose = nusc.get('ego_pose', cam['ego_pose_token'])
                    cam_global2ego = np.eye(4)
                    cam_global2ego[:3, :3] = Quaternion(cam_pose['rotation']).rotation_matrix.T
                    cam_global2ego[:3, 3] = -(cam_global2ego[:3, :3] @ np.array(cam_pose['translation']))
                    ego2cam = np.eye(4)
                    ego2cam[:3, :3] = Quaternion(calib['rotation']).rotation_matrix.T
                    ego2cam[:3, 3] = -(ego2cam[:3, :3] @ np.array(calib['translation']))
                    lidar2cam = ego2cam @ cam_global2ego @ ego2global @ lidar2ego
                    cams[cam_name] = {
                        'token': cam['token'],
                        'img_path': img_path,
                        'timestamp': cam['timestamp'] / 1e6,
                        'cam2img': np.array(calib['camera_intrinsic']),
                        'lidar2cam': lidar2cam
                    }

            lidar_sweep = []
            sweep_token = lidar['prev']
            while sweep_token != '':
                sweep = nusc.get('sample_data', sweep_token)
                if sweep['is_key_frame']:
                    break
                sweep_timestamp = sweep['timestamp'] / 1e6
                sweep_lidar_path = nusc.get_sample_data_path(sweep['token'])
                sweep_pose = nusc.get('ego_pose', sweep['ego_pose_token'])
                sweep_ego2global = np.eye(4)
                sweep_ego2global[:3, :3] = Quaternion(sweep_pose['rotation']).rotation_matrix
                sweep_ego2global[:3, 3] = sweep_pose['translation']
                lidar_sweep.append({
                    'token': sweep['token'],
                    'timestamp': sweep_timestamp,
                    'lidar_path': sweep_lidar_path,
                    'ego2global': sweep_ego2global
                })
                sweep_token = sweep['prev']
            lidar_sweep.reverse()

            labels = []
            _, box_list, _ = nusc.get_sample_data(lidar['token'])
            for box in box_list:
                ann = nusc.get('sample_annotation', box.token)
                labels.append({
                    'token': ann['token'],
                    'name': box.name,
                    'xyz': box.center,
                    'wlh': box.wlh,
                    'quaternion': box.orientation.q,
                    'velocity': box.velocity,
                    'num_lidar_pts': ann['num_lidar_pts']
                })

            scene_samples.append({
                'token': sample['token'],
                'timestamp': timestamp,
                'lidar_path': lidar_path,
                'lidar2ego': lidar2ego,
                'ego2global': ego2global,
                'cams': cams,
                'lidar_sweep': lidar_sweep,
                'labels': labels
            })
            sample_token = sample['next']

        data_info.append({
            'scene_name': scene['name'],
            'samples': scene_samples,
            'description': scene['description']
        })
        print(f"Processed scene: {scene['name']} with {len(scene_samples)} samples.")

    with open(args.save_path, 'wb') as f:
        pickle.dump(data_info, f)
    print(f"Data info saved to {args.save_path}.")