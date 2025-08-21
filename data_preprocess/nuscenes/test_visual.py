import os
import sys
import open3d as o3d
import numpy as np
import pickle

if __name__ == '__main__':
    info_path = './data/nuscenes/nuscenes_data_info.pkl'
    with open(info_path, 'rb') as f:
        data_info = pickle.load(f)

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

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    is_ground = np.fromfile(os.path.join('./data/nuscenes/is_ground', scene['scene_name'], os.path.basename(lidar_path)), dtype=bool)
    print(f"Number of points: {points.shape[0]}, Number of ground points: {np.sum(is_ground)}")

    objects_path = os.path.join('./data/nuscenes/3d_objects', scene['scene_name'], os.path.basename(lidar_path).replace(".bin", ".pkl"))
    with open(objects_path, 'rb') as f:
        objects = pickle.load(f)

    # Visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])
    render_option.point_size = 2
    render_option.line_width = 5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((points.shape[0], 3))
    colors[is_ground, :] = [1, 1, 1]  # White for ground points
    colors[~is_ground, :] = [1, 0, 0]  # Red for non-ground points
    pcd.colors = o3d.utility.Vector3dVector(colors)
    for cluster in objects['cluster_results']:
        if cluster['state'] == 'invalid':
            bbox = cluster['bbox_3d']
            color = [1, 1, 1]
        if cluster['state'] == 'static':
            bbox = cluster['bbox_3d']
            color = [0, 1, 0]
        elif cluster['state'] == 'dynamic':
            bbox = cluster['bbox_3d_fine']
            color = [0, 0, 1]
        bbox[2] = bbox[2] + bbox[5] / 2
        bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                        R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
        pcd_in_box = bbox_geometry.get_point_indices_within_bounding_box(pcd.points)
        # np.asarray(pcd.colors)[pcd_in_box] = np.array(color)
        bbox_geometry.color = color
        vis.add_geometry(bbox_geometry)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("Visualization complete.")
    