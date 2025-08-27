import os
import argparse
import pickle
import numpy as np
import pypatchworkpp


def ground_removal(lidar_path, save_path):
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    points = np.concatenate([points, np.arange(0, len(points))[:, None]], axis=1)
    
    params = pypatchworkpp.Parameters()
    params.verbose = False
    params.enable_RNR = True
    params.sensor_height = 1.723
    params.min_range = 1.0
    params.max_range = 80
    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
    PatchworkPLUSPLUS.estimateGround(points)
    # ground = PatchworkPLUSPLUS.getGround()
    # nonground = PatchworkPLUSPLUS.getNonground()
    ground_idx = np.array(PatchworkPLUSPLUS.getGroundIndices())
    # nonground_idx = PatchworkPLUSPLUS.getNongroundIndices()

    is_ground = np.zeros(points.shape[0], dtype=bool)
    is_ground[ground_idx] = True
    is_ground.tofile(save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--info_path', type=str, default='./data/kitti/kitti_data_info.pkl', help='Path to the Kitti data info file')
    parser.add_argument('--save_dir', type=str, default='./data/kitti/is_ground', help='Directory to save the ground removal results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)
    
    for scene in data_info['kitti_raw'].keys():
        scene_save_dir = f"{args.save_dir}/{scene}"
        os.makedirs(scene_save_dir, exist_ok=True)
        frame_names = list(data_info['kitti_raw'][scene].keys())
        frame_names.sort()

        for name in frame_names:
            lidar_path = data_info['kitti_raw'][scene][name]['lidar_path']
            save_path = os.path.join(scene_save_dir, name + '.bin')
            ground_removal(lidar_path, save_path)
            
        print(f"Processed scene: {scene} with {len(frame_names)} frames.")

    print("Ground removal processing completed.")