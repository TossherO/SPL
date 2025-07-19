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

    # scene_idx = 0
    # sample_idx = 16
    scene_idx = 3
    sample_idx = 10
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
        objects = pickle.load(f)

    points = objects['points_c']
    ground_points = objects['ground_points_c']
    cluster_labels = objects['cluster_labels']

    # Visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])
    render_option.point_size = 3
    render_option.line_width = 5

    colors = np.zeros((points.shape[0], 3))
    cluster_num = cluster_labels.max().item() + 1
    for i in range(cluster_num):
        colors[cluster_labels == i] = np.random.rand(3) * 0.5 + 0.5
    now_frame = points[:, 3] == objects['sample_idx']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[now_frame, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[now_frame])

    ground_colors = np.ones((ground_points.shape[0], 3))
    ground_now_frame = ground_points[:, 3] == objects['sample_idx']
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_points[ground_now_frame, :3])
    ground_pcd.colors = o3d.utility.Vector3dVector(ground_colors[ground_now_frame])

    for cluster in objects['cluster_results']:
        if cluster['state'] == 'invalid':
            continue
            # bbox = cluster['bbox_3d']
            # color = [1, 1, 1]
        if cluster['state'] == 'static':
            bbox = cluster['bbox_3d']
            color = [0, 1, 0]
        elif cluster['state'] == 'dynamic':
            bbox = cluster['bbox_3d_fine']
            color = [0, 1, 0]
        bbox[2] = bbox[2] + bbox[5] / 2
        bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                        R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
        bbox_geometry.color = color
        vis.add_geometry(bbox_geometry)

    for label in sample['labels']:
        if label['name'].split('.')[0] not in ['human', 'vehicle']:
            continue
        x, y, z = label['xyz']
        w, l, h = label['wlh']
        R = Quaternion(label['quaternion']).rotation_matrix
        bbox_geometry = o3d.geometry.OrientedBoundingBox(
            center=[x, y, z],
            extent= [l, w, h],
            R=R
        )
        bbox_geometry.color = [1, 0, 0]
        vis.add_geometry(bbox_geometry)

    vis.add_geometry(pcd)
    vis.add_geometry(ground_pcd)
    vis.run()
    vis.destroy_window()
    print("Visualization complete.")
    