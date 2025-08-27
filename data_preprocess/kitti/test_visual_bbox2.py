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

    for scene in data_info['kitti_raw'].keys():
        raw_data = data_info['kitti_raw'][scene]
        pseudo_labels_path = f'./data/kitti/pseudo_labels/{scene}.pkl'
        with open(pseudo_labels_path, 'rb') as f:
            pseudo_labels = pickle.load(f)

        frame_names = list(raw_data.keys())
        frame_names.sort()
        for name in frame_names:
            points = np.fromfile(raw_data[name]['lidar_path'], dtype=np.float32).reshape(-1, 4)[:, :3]
            is_ground = np.fromfile(os.path.join('./data/kitti/is_ground', scene, name + '.bin'), dtype=bool)
            ground_points = points[is_ground]
            points = points[~is_ground]
            objects = pseudo_labels[name]['pseudo_labels_3d']

            # Visualize the point cloud
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            render_option = vis.get_render_option()
            render_option.background_color = np.array([0, 0, 0])
            render_option.point_size = 2
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
                    if obj['name'] == 'car':
                        color = [0, score * 2, 0]
                    elif obj['name'] == 'pedestrian':
                        color = [0, 0, score * 2]
                    else:
                        color = [0, score * 2, score * 2]
                    bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                                    R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
                    bbox_geometry.color = color
                    pcd_in_box = bbox_geometry.get_point_indices_within_bounding_box(pcd.points)
                    np.asarray(pcd.colors)[pcd_in_box] = np.array(color)
                    vis.add_geometry(bbox_geometry)

                    if sum(obj['vel_3d'] ** 2) > 1:
                        arrow_start = bbox[:3]
                        l = bbox[3] * 0.5 if bbox[3] > 1.0 else 0.5
                        arrow_end = arrow_start + np.array([l * np.cos(bbox[6]), l * np.sin(bbox[6]), 0])
                        arrow = o3d.geometry.LineSet()
                        arrow.points = o3d.utility.Vector3dVector([arrow_start, arrow_end])
                        arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
                        arrow.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
                        vis.add_geometry(arrow)

                    if obj.get('max_lwh', None) is not None:
                        max_bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=obj['max_lwh'], 
                                            R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
                        max_bbox_geometry.color = [0, 0, 1]
                        vis.add_geometry(max_bbox_geometry)

            vis.add_geometry(pcd)
            vis.add_geometry(ground_pcd)
            vis.run()
            vis.destroy_window()
            print("Visualization complete.")
        