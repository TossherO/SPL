import os
import argparse
import pickle
import numpy as np
import torch


def load_ground_points(lidar_with_sweeps, sample_idx, scene_save_dir):
    ground_points_c = []
    start = max(0, sample_idx - (args.frame_len - 1) // 2)
    end = min(len(lidar_with_sweeps) - 1, sample_idx + (args.frame_len + 1) // 2)
    for i in range(start, end + 1):
        lidar_path = lidar_with_sweeps[i]['lidar_path']
        lidar2global = torch.tensor(lidar_with_sweeps[i]['lidar2global'], dtype=torch.float32).cuda()
        timestamp = lidar_with_sweeps[i]['timestamp']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        points = torch.tensor(points, dtype=torch.float32).cuda()
        is_ground = np.fromfile(os.path.join(args.is_ground_dir, os.path.basename(scene_save_dir), os.path.basename(lidar_path)), dtype=bool)
        is_ground = torch.tensor(is_ground, dtype=torch.bool).cuda()
        R = lidar2global[:3, :3]
        T = lidar2global[:3, 3]
        points = (R @ points.T).T + T
        ground_points = points[is_ground]
        ground_points_c.append(ground_points)
    ground_points_c = torch.cat(ground_points_c, dim=0)
    lidar2global = torch.tensor(lidar_with_sweeps[sample_idx]['lidar2global'], dtype=torch.float32).cuda()
    R_inv = lidar2global[:3, :3].T
    T_inv = -R_inv @ lidar2global[:3, 3]
    ground_points_c = (R_inv @ ground_points_c.T).T + T_inv
    return ground_points_c


def fit_bounding_box(points):
    """
    CUDA并行优化的L-shape框拟合算法（修正版）
    
    Args:
        cluster_points (torch.Tensor): 点云数据 (N, 4)
    
    Returns:
        T_reference_bbox (torch.Tensor): 4x4变换矩阵
        bboxdimensions (list): [长度, 宽度, 高度]
    """
    device = points.device
    
    # 1. 并行角度搜索
    delta = 1
    angles_deg = torch.arange(0, 91, delta, device=device)  # 0-90度
    angles_rad = torch.deg2rad(angles_deg)
    
    # 批量生成旋转矩阵 [91, 2, 2]
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    R_matrices = torch.stack([
        torch.stack([cos_a, sin_a], dim=1),
        torch.stack([-sin_a, cos_a], dim=1)
    ], dim=1)
    
    # 点云数据准备 [N, 2] -> [1, N, 2]
    xy_points = points[:, :2].unsqueeze(0)
    
    # 批量旋转点云 [91, N, 2]
    rotated_points = torch.matmul(R_matrices, xy_points.transpose(1, 2)).transpose(1, 2)
    
    # 并行计算紧凑性指标
    min_x = rotated_points[:, :, 0].min(dim=1).values
    max_x = rotated_points[:, :, 0].max(dim=1).values
    min_y = rotated_points[:, :, 1].min(dim=1).values
    max_y = rotated_points[:, :, 1].max(dim=1).values
    
    # 计算点到边界的距离
    Dx = torch.min(rotated_points[:, :, 0] - min_x.unsqueeze(1), max_x.unsqueeze(1) - rotated_points[:, :, 0])
    Dy = torch.min(rotated_points[:, :, 1] - min_y.unsqueeze(1), max_y.unsqueeze(1) - rotated_points[:, :, 1])
    
    # 计算紧凑性指标beta [91]
    d0 = torch.tensor(1e-2, device=device)
    beta = torch.min(Dx, Dy).clamp(min=d0)
    beta = (1 / beta).sum(dim=1)
    
    # 选择最佳角度
    best_idx = torch.argmax(beta)
    choose_angle = angles_rad[best_idx]
    
    # 2. 提取最优角度对应的边界值（关键修正）
    min_x_val = min_x[best_idx]
    max_x_val = max_x[best_idx]
    min_y_val = min_y[best_idx]
    max_y_val = max_y[best_idx]
    range_x = max_x_val - min_x_val
    range_y = max_y_val - min_y_val
    
    # 3. 方向校正（确保长边对齐）
    if range_x < range_y:
        choose_angle += np.pi / 2
        R_best = torch.tensor([
            [torch.cos(choose_angle), torch.sin(choose_angle)],
            [-torch.sin(choose_angle), torch.cos(choose_angle)]
        ], device=device)
        rotated_best = (R_best @ points[:, :2].T).T
        min_x_val = rotated_best[:, 0].min()
        max_x_val = rotated_best[:, 0].max()
        min_y_val = rotated_best[:, 1].min()
        max_y_val = rotated_best[:, 1].max()
    else:
        R_best = R_matrices[best_idx]
    
    # 4. 边界框构造
    corners_local = torch.tensor([
        [max_x_val, min_y_val],
        [min_x_val, min_y_val],
        [min_x_val, max_y_val],
        [max_x_val, max_y_val]
    ], device=device)
    
    corners_ref = (R_best.T @ corners_local.T).T
    
    # 5. 中心点和变换矩阵
    bbox_center_ref = torch.zeros(3, device=device)
    bbox_center_ref[0] = corners_ref[:, 0].mean()
    bbox_center_ref[1] = corners_ref[:, 1].mean()
    bbox_center_ref[2] = points[:, 2].min()
    
    T_matrix = torch.eye(4, device=device)
    T_matrix[:3, 3] = bbox_center_ref
    T_matrix[:2, :2] = R_best.T
    
    # 6. 尺寸计算
    length = torch.norm(corners_ref[1] - corners_ref[0])
    width = torch.norm(corners_ref[3] - corners_ref[0])
    height = points[:, 2].max() - points[:, 2].min()
    
    return T_matrix, [length.item(), width.item(), height.item()]


def process_3d_objects(objects, points, ground_points, lidar2imgs):
    for i, obj in enumerate(objects):
        if obj['category'] == 'car':
            points_in_obj = points[(points[:, 3] == i) & (points[:, 4] == obj['timestamp'])]
            if len(points_in_obj) < 10:
                continue
            
            T_matrix, bbox_dimensions = fit_bounding_box(points_in_obj)
            obj['T_matrix'] = T_matrix.cpu().numpy()
            obj['bbox_dimensions'] = bbox_dimensions
            
            # Project points to images
            for j, lidar2img in enumerate(lidar2imgs):
                projected_points = (lidar2img @ T_matrix @ torch.cat([points_in_obj[:, :3], torch.ones((points_in_obj.shape[0], 1), device=points.device)], dim=1).T).T
                obj[f'projected_points_cam{j}'] = projected_points[:, :2].cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--frame_len', type=int, default=11, help='Number of frames to process at once')
    parser.add_argument('--img_hw', type=int, nargs=2, default=[900, 1600], help='Image width and height for projection')
    parser.add_argument('--info_path', type=str, default='./data/nuscenes/nuscenes_data_info.pkl', help='Path to the NuScenes data info file')
    parser.add_argument('--is_ground_dir', type=str, default='./data/nuscenes/is_ground', help='Directory containing ground removal results')
    parser.add_argument('--objects_dir', type=str, default='./data/nuscenes/3d_objects', help='Directory containing 3D object results')
    parser.add_argument('--save_dir', type=str, default='./data/nuscenes/pseudo_labels', help='Directory to save the processed pseudo labels')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)

    for scene in data_info:
        scene_name = scene['scene_name']
        scene_save_dir = f"{args.save_dir}/{scene_name}"

        lidar_with_sweeps = []
        sample_idx_list = []
        count = 0
        for sample in scene['samples']:
            for sweep in sample['lidar_sweep']:
                lidar_with_sweeps.append({
                    'timestamp': sweep['timestamp'],
                    'lidar_path': sweep['lidar_path'],
                    'lidar2global': sweep['ego2global'] @ sample['lidar2ego'],
                })
                count += 1
            lidar_with_sweeps.append({
                'timestamp': sample['timestamp'],
                'lidar_path': sample['lidar_path'],
                'lidar2global': sample['ego2global'] @ sample['lidar2ego'],
                'cams': sample['cams']
            })
            sample_idx_list.append(count)
            count += 1

        scene_results = []
        with torch.no_grad():
            for sample_idx in sample_idx_list:
                lidar_path = lidar_with_sweeps[sample_idx]['lidar_path']
                object_path = os.path.join(args.objects_dir, scene_name, os.path.basename(lidar_path).replace('.bin', '.pkl'))
                
                with open(object_path, 'rb') as f:
                    objects_info = pickle.load(f)
                objects = objects_info['objects']
                points = torch.tensor(objects_info['points'], dtype=torch.float32).cuda()
                ground_points = load_ground_points(lidar_with_sweeps, sample_idx, scene_save_dir)
                
                cam_infos = lidar_with_sweeps[sample_idx]['cams']
                lidar2imgs = []
                for cam in cam_infos:
                    cam2img = np.eye(4, dtype=np.float32)
                    cam2img[:3, :3] = cam_infos[cam]['cam2img'][:3, :3]
                    lidar2imgs.append(cam2img @ cam_infos[cam]['lidar2cam'])
                lidar2imgs = torch.tensor(np.array(lidar2imgs), dtype=torch.float32).cuda()

                results = process_3d_objects(objects, points, ground_points, lidar2imgs)
                scene_results.append({
                    'timestamp': lidar_with_sweeps[sample_idx]['timestamp'],
                    'lidar2global': lidar_with_sweeps[sample_idx]['lidar2global'],
                    'objects': results,
                })
