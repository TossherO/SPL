import os
import sys
import open3d as o3d
import numpy as np
import pickle
import cv2

if __name__ == '__main__':
    info_path = './data/kitti/kitti_data_info.pkl'
    with open(info_path, 'rb') as f:
        data_info = pickle.load(f)

    sample_idx = 0
    sample = data_info['kitti_object'][sample_idx]
    raw_data = data_info['kitti_raw'][sample['scene']]

    # lidar_path = raw_data[str(sample['sample_idx']).zfill(10)]['lidar_path']
    # points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    # is_ground = np.fromfile(os.path.join('./data/kitti/is_ground', sample['name'] + '.bin'), dtype=bool)
    # print(f"Number of points: {points.shape[0]}, Number of ground points: {np.sum(is_ground)}")

    # Visualize the point cloud
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # render_option = vis.get_render_option()
    # render_option.background_color = np.array([0, 0, 0])
    # render_option.point_size = 2
    # render_option.line_width = 5
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # colors = np.zeros((points.shape[0], 3))
    # colors[is_ground, :] = [1, 1, 1]  # White for ground points
    # colors[~is_ground, :] = [1, 0, 0]  # Red for non-ground points
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # vis.add_geometry(pcd)
    # vis.run()
    # vis.destroy_window()
    # print("Visualization complete.")

    # img_path = raw_data[str(sample['sample_idx']).zfill(10)]['img_path']
    # img = cv2.imread(img_path)
    # lidar2cam = sample['lidar2cam']
    # cam2img = sample['cam2img']
    # lidar2img = cam2img @ lidar2cam
    # points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    # points_img = (lidar2img @ points_hom.T).T
    # points_img[:, :2] = points_img[:, :2] / np.clip(points_img[:, 2:3], a_min=1e-4, a_max=None)
    # mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < img.shape[1]) & (points_img[:, 1] >= 0) & (points_img[:, 1] < img.shape[0]) & (points_img[:, 2] > 0)
    # points_img = points_img[mask]
    # points_img = points_img[:, :2].astype(np.int32)
    # is_ground = is_ground[mask]
    # img_vis = img.copy()
    # img_vis[points_img[is_ground, 1], points_img[is_ground, 0], :] = [0, 255, 0]  # Green for ground points
    # img_vis[points_img[~is_ground, 1], points_img[~is_ground, 0], :] = [0, 0, 255]  # Red for
    # cv2.imshow('Lidar Projection', img_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    ego2lidar = sample['ego2lidar']
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = ego2lidar[:3, :3].T
    lidar2ego[:3, 3] = -ego2lidar[:3, :3].T @ ego2lidar[:3, 3]
    points_c = []
    for i in range(sample['start_idx'], sample['end_idx'] + 1):
        idx = str(i).zfill(10)
        lidar_path = raw_data[idx]['lidar_path']
        lidar2global = raw_data[idx]['lidar2global']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        points = (lidar2global[:3, :3] @ points.T).T + lidar2global[:3, 3]
        points_c.append(points)
    lidar2global = raw_data[str(sample['sample_idx']).zfill(10)]['lidar2global']
    global2lidar = np.eye(4)
    global2lidar[:3, :3] = lidar2global[:3, :3].T
    global2lidar[:3, 3] = -lidar2global[:3, :3].T @ lidar2global[:3, 3]
    points_c = np.concatenate(points_c, axis=0, dtype=np.float32)
    points_c = (global2lidar[:3, :3] @ points_c.T).T + global2lidar[:3, 3]

    # Visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])
    render_option.point_size = 1.5
    render_option.line_width = 5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_c)
    colors = np.ones((points_c.shape[0], 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("Visualization complete.")