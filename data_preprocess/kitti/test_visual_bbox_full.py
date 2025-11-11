import os
import sys
import open3d as o3d
import numpy as np
import pickle
from pyquaternion import Quaternion


if __name__ == '__main__':
    info_path = './data/kitti/kitti_data_info.pkl'
    info_path2 = './pcdet/data/kitti/kitti_infos_train.pkl'
    with open(info_path, 'rb') as f:
        data_info = pickle.load(f)
    with open(info_path2, 'rb') as f:
        train_info = pickle.load(f)

    gt_colors = {
        'Car': [1, 0, 0],
        'Pedestrian': [0, 1, 0],
        'Cyclist': [0, 0, 1],
    }
    pseudo_colors = {
        'Car': [1, 1, 0],
        'Pedestrian': [0, 1, 1],
        'Cyclist': [1, 0, 1],
    }
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=False)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])
    render_option.point_size = 1.0
    render_option.line_width = 5.0
    ctr = vis.get_view_control()

    for sample_idx in range(0, 3000, 100):
        sample = data_info['kitti_object'][sample_idx]
        raw_data = data_info['kitti_raw'][sample['scene']]
        lidar_path = os.path.join('./data/kitti/training/velodyne', train_info[sample_idx]['point_cloud']['lidar_idx'] + '.bin')
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        mask = ((points[:, 0] >= point_cloud_range[0]) & (points[:, 0] <= point_cloud_range[3]) &
                (points[:, 1] >= point_cloud_range[1]) & (points[:, 1] <= point_cloud_range[4]) &
                (points[:, 2] >= point_cloud_range[2]) & (points[:, 2] <= point_cloud_range[5]))
        points = points[mask]

        pseudo_labels_path = f'./data/kitti/pseudo_labels/{sample["scene"]}.pkl'
        with open(pseudo_labels_path, 'rb') as f:
            pseudo_labels = pickle.load(f)
        label = pseudo_labels[str(sample['sample_idx']).zfill(10)]
        objects = label['pseudo_labels_3d']

        gt_names = train_info[sample_idx]['annos']['name']
        gt_bboxes = train_info[sample_idx]['annos']['gt_boxes_lidar']

        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1.0, 1.0, 1.0])

        for obj in objects:
            if obj['bbox_3d'] is not None:
                bbox = obj['bbox_3d']
                score = obj['score_3d']
                bbox[2] = bbox[2] + bbox[5] / 2
                color = pseudo_colors[obj['name']]
                color = [c * min(1.0, score + 0.2) for c in color]
                bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                                R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
                bbox_geometry.color = color
                # pcd_in_box = bbox_geometry.get_point_indices_within_bounding_box(pcd.points)
                # np.asarray(pcd.colors)[pcd_in_box] = np.array(color)
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

                # if obj.get('max_lwh', None) is not None:
                #     max_bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=obj['max_lwh'], 
                #                         R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
                #     max_bbox_geometry.color = [0, 0, 1]
                #     vis.add_geometry(max_bbox_geometry)

        for gt_idx in range(gt_bboxes.shape[0]):
            if gt_names[gt_idx] not in gt_colors:
                continue
            bbox = gt_bboxes[gt_idx]
            color = gt_colors[gt_names[gt_idx]]
            bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                            R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
            bbox_geometry.color = color
            pcd_in_box = bbox_geometry.get_point_indices_within_bounding_box(pcd.points)
            np.asarray(pcd.colors)[pcd_in_box] = np.array(color)
            vis.add_geometry(bbox_geometry)

        vis.add_geometry(pcd)

        ctr.set_lookat([20, 0, 0])
        ctr.set_zoom(0.3)
        ctr.set_front([-1, 0, 1])
        ctr.set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()

        if not os.path.exists('./output'):
            os.makedirs('./output')
        vis.capture_screen_image('./output/' + str(sample_idx) + '.png', do_render=True)
    
    vis.destroy_window()