import os
import sys
import open3d as o3d
import numpy as np
import pickle
import cv2

if __name__ == '__main__':

    points = np.fromfile('./006431.bin', dtype=np.float32).reshape(-1, 4)[:, :3]
    with open('./objects3227.pkl', 'rb') as f:
        objects = pickle.load(f)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])
    render_option.point_size = 5
    render_option.line_width = 5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1.0, 1.0, 1.0])

    colors = {
        'Car': [1, 0, 0],
        'Pedestrian': [0, 1, 0],
        'Cyclist': [0, 0, 1],
    }
    for obj in objects:
        if obj['bbox_3d'] is not None:
            if obj['name'] != 'Cyclist':
                continue
            bbox = obj['bbox_3d']
            score = obj['score_3d']
            bbox[2] = bbox[2] + bbox[5] / 2
            color = colors[obj['name']]
            # color = [c * min(1.0, score + 0.2) for c in color]
            bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                            R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
            bbox_geometry.color = color
            pcd_in_box = bbox_geometry.get_point_indices_within_bounding_box(pcd.points)
            np.asarray(pcd.colors)[pcd_in_box] = np.array(color)
            vis.add_geometry(bbox_geometry)

            # if sum(obj['vel_3d'] ** 2) > 1:
            #     arrow_start = bbox[:3]
            #     l = bbox[3] * 0.5 if bbox[3] > 1.0 else 0.5
            #     arrow_end = arrow_start + np.array([l * np.cos(bbox[6]), l * np.sin(bbox[6]), 0])
            #     arrow = o3d.geometry.LineSet()
            #     arrow.points = o3d.utility.Vector3dVector([arrow_start, arrow_end])
            #     arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
            #     arrow.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
            #     vis.add_geometry(arrow)

            # if obj.get('max_lwh', None) is not None:
            #     max_bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=obj['max_lwh'], 
            #                         R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
            #     max_bbox_geometry.color = [0, 0, 1]
            #     vis.add_geometry(max_bbox_geometry)

    vis.add_geometry(pcd)
    vis.run()
    # vis.capture_screen_image('./output_obj2.png')
    vis.destroy_window()
    print("Visualization complete.")