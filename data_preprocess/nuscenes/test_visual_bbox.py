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

    scene_idx = 0
    scene = data_info[scene_idx]
    scene_name = scene['scene_name']
    pseudo_labels_path = f'./data/nuscenes/pseudo_labels/{scene_name}.pkl'
    with open(pseudo_labels_path, 'rb') as f:
        pseudo_labels = pickle.load(f)

    for sample_idx in range(0, len(data_info[scene_idx]['samples']), 5):
        sample = scene['samples'][sample_idx]
        lidar_path = sample['lidar_path']
        objects = pseudo_labels[sample_idx]['objects']

        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        is_ground = np.fromfile(os.path.join('./data/nuscenes/is_ground', scene['scene_name'], os.path.basename(lidar_path)), dtype=bool)
        ground_points = points[is_ground]
        points = points[~is_ground]

        # Visualize the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0, 0, 0])
        render_option.point_size = 2.5
        render_option.line_width = 5

        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = 1.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        ground_colors = np.ones((ground_points.shape[0], 3))
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
        ground_pcd.colors = o3d.utility.Vector3dVector(ground_colors)

        for obj in objects:
            if obj['bbox_3d'] is not None:
                bbox = obj['bbox_3d']
                score = obj['score_3d']
                bbox[2] = bbox[2] + bbox[5] / 2
                bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                                R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
                color = [0, score * 2, 0]
                bbox_geometry.color = color
                pcd_in_box = bbox_geometry.get_point_indices_within_bounding_box(pcd.points)
                np.asarray(pcd.colors)[pcd_in_box] = np.array(color)
                vis.add_geometry(bbox_geometry)

        vis.add_geometry(pcd)
        vis.add_geometry(ground_pcd)
        vis.run()
        vis.destroy_window()
        print("Visualization complete.")
    