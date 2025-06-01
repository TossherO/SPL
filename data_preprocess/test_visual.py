import os
import sys
import open3d as o3d
import numpy as np
import pickle

if __name__ == '__main__':
    info_path = './data/nuscenes/nuscenes_data_info.pkl'
    with open(info_path, 'rb') as f:
        data_info = pickle.load(f)

    scene_idx = 2
    sample_idx = 39
    sweep_idx = -1
    scene = data_info[scene_idx]
    sample = scene['samples'][sample_idx]
    print(f"Scene: {scene['scene_name']}, Sample: {sample_idx}, Sweep: {sweep_idx}")

    if sweep_idx >= 0:
        lidar_path = sample['lidar_sweep'][sweep_idx]['lidar_path']
    else:
        lidar_path = sample['lidar_path']

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    
    is_ground = np.fromfile(os.path.join('./data/nuscenes/is_ground', scene['scene_name'], os.path.basename(lidar_path)), dtype=bool)
    
    print(f"Number of points: {points.shape[0]}, Number of ground points: {np.sum(is_ground)}")

    # Visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((points.shape[0], 3))
    colors[is_ground, :] = [0, 1, 0]  # Green for ground points
    colors[~is_ground, :] = [1, 0, 0]  # Red for non-ground points
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("Visualization complete.")
    