import os
import sys
import open3d as o3d
import numpy as np
import pickle
from pyquaternion import Quaternion


if __name__ == '__main__':
    info_path = './data/nuscenes/nuscenes_data_info.pkl'
    with open(info_path, 'rb') as f:
        data_info = pickle.load(f)

    scene_idx = 9
    for sample_idx in range(0, len(data_info[scene_idx]['samples']), 5):
    # scene_idx = 3
    # sample_idx = 10
        sweep_idx = -1
        scene = data_info[scene_idx]
        sample = scene['samples'][sample_idx]
        print(f"Scene: {scene['scene_name']}, Sample: {sample_idx}, Sweep: {sweep_idx}")

        if sweep_idx >= 0:
            lidar_path = sample['lidar_sweep'][sweep_idx]['lidar_path']
        else:
            lidar_path = sample['lidar_path']

        objects_path = os.path.join('./data/nuscenes/3d_objects', scene['scene_name'], os.path.basename(lidar_path).replace(".bin", ".pkl"))
        with open(objects_path, 'rb') as f:
            results = pickle.load(f)

        # points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        # is_ground = np.fromfile(os.path.join('./data/nuscenes/is_ground', scene['scene_name'], os.path.basename(lidar_path)), dtype=bool)
        # ground_points = points[is_ground]
        # points = points[~is_ground]

        # points_path = os.path.join('./data/nuscenes/3d_objects', scene['scene_name'], os.path.basename(lidar_path))
        # points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 3)
        objects = results['objects']
        points = results['points'][:, :3]

        # Visualize the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0, 0, 0])
        render_option.point_size = 2.5
        render_option.line_width = 5

        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = 1.0  # Set red channel to 1 for all points
        for now_object in objects:
            if now_object['points'] is None or now_object['points'].shape[0] == 0:
                continue
            colors[now_object['points']] = np.random.rand(3)*0.5 + 0.5  # Random color for each object

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # ground_colors = np.ones((ground_points.shape[0], 3))
        # ground_pcd = o3d.geometry.PointCloud()
        # ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
        # ground_pcd.colors = o3d.utility.Vector3dVector(ground_colors)

        vis.add_geometry(pcd)
        # vis.add_geometry(ground_pcd)
        vis.run()
        vis.destroy_window()
        print("Visualization complete.")
    