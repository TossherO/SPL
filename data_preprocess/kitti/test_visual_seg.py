import os
import sys
import open3d as o3d
import numpy as np
import pickle
from pyquaternion import Quaternion


if __name__ == '__main__':
    info_path = './data/kitti/kitti_data_info.pkl'
    with open(info_path, 'rb') as f:
        data_info = pickle.load(f)

    for sample in data_info['kitti_object']:
        raw_data = data_info['kitti_raw'][sample['scene']]
        name = str(sample['sample_idx']).zfill(10)
        objects_path = os.path.join('./data/kitti/3d_objects', sample['scene'], name + '.pkl')
        with open(objects_path, 'rb') as f:
            objects = pickle.load(f)['objects']

        points = np.fromfile(raw_data[name]['lidar_path'], dtype=np.float32).reshape(-1, 4)[:, :3]
        is_ground = np.fromfile(os.path.join('./data/kitti/is_ground', sample['scene'], name + '.bin'), dtype=bool)
        points = points[~is_ground]
        pc_range = np.array([0, -40.0, -3.0, 70.4, 40.0, 1.0], dtype=np.float32).reshape(2, 3)
        mask = np.all((points[:, :3] >= pc_range[0]) & (points[:, :3] <= pc_range[1]), axis=1)
        points = points[mask]

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
    