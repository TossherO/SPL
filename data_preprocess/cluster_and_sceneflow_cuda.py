import os
import argparse
import pickle
import numpy as np
import torch
import hdbscan
import open3d
import scipy
from typing import Tuple
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def fit_bounding_box(cluster_points):
    """
    CUDA并行优化的L-shape框拟合算法（修正版）
    
    Args:
        cluster_points (torch.Tensor): 点云数据 (N, 4)
    
    Returns:
        T_reference_bbox (torch.Tensor): 4x4变换矩阵
        bboxdimensions (list): [长度, 宽度, 高度]
    """
    device = cluster_points.device
    
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
    xy_points = cluster_points[:, :2].unsqueeze(0)
    
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
        rotated_best = (R_best @ cluster_points[:, :2].T).T
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
    bbox_center_ref[2] = cluster_points[:, 2].min()
    
    T_matrix = torch.eye(4, device=device)
    T_matrix[:3, 3] = bbox_center_ref
    T_matrix[:2, :2] = R_best.T
    
    # 6. 尺寸计算
    length = torch.norm(corners_ref[1] - corners_ref[0])
    width = torch.norm(corners_ref[3] - corners_ref[0])
    height = cluster_points[:, 2].max() - cluster_points[:, 2].min()
    
    return T_matrix, [length.item(), width.item(), height.item()]


def get_histogram_based_and_icp_based_transformations(source_points:torch.Tensor, target_points:torch.Tensor, search_size:float, search_step:float, max_icp_iterations:int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute histogram-based and ICP-based homogeneous transformation matrices.
    
    Args:
        source_points (torch.Tensor) : Source LiDAR point cloud with shape (N,4).
        target_points (torch.Tensor) : Target LiDAR point cloud with shape (N,4).
        search_size (float) : Search size in x and y direction in meters.
        search_step (float) : Step size in x and y direction in meters.
        max_icp_iterations (int) : Maximum number of iterations for ICP.
        
    Returns:
        T_hist (torch.Tensor) : Histogram-based homogeneous transformation matrix.
        T_icp_3dof (torch.Tensor) : ICP-based homogeneous transformation matrix.
    """
    def unravel_index(index, shape):   # Torch 1.x has no torch.unravel_index.
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))
    
    x_values = torch.linspace(-search_size, search_size, int(2*search_size/search_step+1))
    y_values = torch.linspace(-search_size, search_size, int(2*search_size/search_step+1))
    z_values = torch.linspace(-0.10, 0.10, 11)
    
    translation_vectors = (target_points[:,:3].reshape(1,-1,3)-source_points[:,:3].reshape(-1,1,3))
    
    H, _ = torch.histogramdd(translation_vectors.reshape(-1,3), bins=(x_values, y_values, z_values))
    x_idx, y_idx, z_idx = unravel_index(torch.argmax(H).item(), H.shape)
    
    best_x = (x_values[x_idx]+x_values[x_idx+1])/2
    best_y = (y_values[y_idx]+y_values[y_idx+1])/2
    best_z = (z_values[z_idx]+z_values[z_idx+1])/2
    
    T_hist = torch.eye(4)
    T_hist[0,3] = best_x
    T_hist[1,3] = best_y
    T_hist[2,3] = best_z
    
    source = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_points[:,:3])
    target = open3d.geometry.PointCloud()
    target.points = open3d.utility.Vector3dVector(target_points[:,:3])
    reg = open3d.pipelines.registration.registration_icp(source=source,
                                                         target=target,
                                                         max_correspondence_distance=2*search_step,
                                                         init=T_hist.clone(),
                                                         estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                         criteria=open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_icp_iterations))
    T_icp = torch.from_numpy(reg.transformation.copy())   # T_t1_t2 := t2 pose in t1 coordinate system.
    
    x_icp, y_icp, z_icp = T_icp[:3,3]
    yaw_radians = torch.arctan2(T_icp[1,0], T_icp[0,0])
    
    T_icp3dof = torch.eye(4)
    T_icp3dof[0,3]   = x_icp
    T_icp3dof[1,3]   = y_icp
    T_icp3dof[2,3]   = z_icp
    T_icp3dof[:2,:2] = torch.tensor([[torch.cos(yaw_radians), -torch.sin(yaw_radians)], [torch.sin(yaw_radians), torch.cos(yaw_radians)]])
    
    return T_hist, T_icp3dof


def cluster_and_sceneflow(lidar_with_sweeps, sample_idx, scene_save_dir):
    # Load the point cloud data
    time1 = time.time()
    points_c = []
    ground_points_c = []
    start = max(0, sample_idx - (args.frame_len - 1) // 2)
    end = min(len(lidar_with_sweeps), sample_idx + (args.frame_len + 1) // 2)
    for i in range(start, end):
        lidar_path = lidar_with_sweeps[i]['lidar_path']
        lidar2global = lidar_with_sweeps[i]['lidar2global']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        is_ground = np.fromfile(os.path.join(args.is_ground_dir, os.path.basename(scene_save_dir), os.path.basename(lidar_path)), dtype=bool)
        R = lidar2global[:3, :3]
        T = lidar2global[:3, 3]
        points = (R @ points.T).T + T
        points = np.concatenate([points, np.ones((points.shape[0], 1)) * i], axis=1)
        ground_points = points[is_ground]
        points = points[~is_ground]
        points_c.append(points)
        ground_points_c.append(ground_points)
    points_c = np.concatenate(points_c, axis=0, dtype=np.float32)
    ground_points_c = np.concatenate(ground_points_c, axis=0, dtype=np.float32)
    lidar2global = lidar_with_sweeps[sample_idx]['lidar2global']
    R_inv = lidar2global[:3, :3].T
    T_inv = -R_inv @ lidar2global[:3, 3]
    points_c[:, :3] = ((R_inv @ points_c[:, :3].T).T + T_inv).astype(np.float32)
    ground_points_c[:, :3] = ((R_inv @ ground_points_c[:, :3].T).T + T_inv).astype(np.float32)

    # Filter points based on the point cloud range
    pc_range = np.array(args.pc_range, dtype=np.float32).reshape(2, 3)
    mask = np.all((points_c[:, :3] >= pc_range[0]) & (points_c[:, :3] <= pc_range[1]), axis=1)
    points_c = points_c[mask]

    time2 = time.time()
    # Perform clustering using HDBSCAN
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=args.cluster_min_size, cluster_selection_epsilon=args.cluster_selection_epsilon)
    points_for_clustering = points_c[:, :3].copy()
    # points_for_clustering[:, 3] = points_for_clustering[:, 3] / 10
    cluster_labels = hdbscan_clusterer.fit_predict(points_for_clustering)
    
    time3 = time.time()
    # Fit bounding boxes to clusters and filter based on size, ratio and area
    points_c = torch.tensor(points_c, dtype=torch.float32).cuda()
    ground_points_c = torch.tensor(ground_points_c, dtype=torch.float32).cuda()
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.int32).cuda()
    cluster_num = cluster_labels.max().item() + 1
    cluster_results = []
    for i in range(cluster_num):
        # Fit bounding box to the cluster
        cluster_point_indices = torch.where(cluster_labels == i)[0]
        cluster_points = points_c[cluster_point_indices]
        T_reference_bbox, bboxdimensions = fit_bounding_box(cluster_points)
        x, y, z = T_reference_bbox[:3, 3].tolist()
        l, w, h = bboxdimensions
        yaw = torch.atan2(T_reference_bbox[1, 0], T_reference_bbox[0, 0]).item()

        # 取bbox附近地面点的平均高度作为bbox的底部高度
        R_inv = T_reference_bbox[:3, :3].T
        T_inv = -R_inv @ T_reference_bbox[:3, 3]
        ground_points_r = (R_inv @ ground_points_c[:, :3].T).T + T_inv
        ground_points_r = ground_points_r[(abs(ground_points_r[:, 0] - x) < l / 2) &
                                          (abs(ground_points_r[:, 1] - y) < w / 2)]
        height_above_ground = 0
        if ground_points_r.shape[0] > 10:
            height_above_ground = ground_points_r[:, 2].mean()
            z = z - height_above_ground
            h = h + height_above_ground

        # Filter clusters based on size, ratio and area
        if height_above_ground > args.cluster_max_z_above_ground:
            continue
        if (l < args.cluster_size_range[0] or l > args.cluster_size_range[3] or
            w < args.cluster_size_range[1] or w > args.cluster_size_range[4] or
            h < args.cluster_size_range[2] or h > args.cluster_size_range[5]):
            continue
        if l * w < args.cluster_min_area or l / w > args.cluster_max_ratio:
            continue
        cluster_results.append({
            'points': cluster_points[cluster_points[:, 3] == sample_idx, :3],
            'cluster_point_indices': cluster_point_indices,
            'bbox_3d': [x, y, z, l, w, h, yaw],
        })
    
    time4 = time.time()
    # Sceneflow estimation for each cluster
    for cluster_result in cluster_results:
        cluster_point_indices = cluster_result['cluster_point_indices']
        cluster_points = points_c[cluster_point_indices]
        sample_points = cluster_result['points']
        if sample_points.shape[0] < args.sceneflow_min_size:
            cluster_result['state'] = 'invalid'
            continue

        # Select the first and last frame indices for the sceneflow estimation
        first_idx = last_idx = None
        for i in range(start, sample_idx):
            _first_points = first_points = cluster_points[cluster_points[:, 3] == i]
            if first_points.shape[0] > args.sceneflow_min_size:
                first_idx = i
                if first_points.shape[0] > args.sceneflow_max_size:
                    torch.manual_seed(0)
                    _first_points = first_points[torch.randperm(first_points.shape[0])[:args.sceneflow_max_size]]
                break
        for i in range(end - 1, sample_idx, -1):
            _last_points = last_points = cluster_points[cluster_points[:, 3] == i]
            if last_points.shape[0] > args.sceneflow_min_size:
                last_idx = i
                if last_points.shape[0] > args.sceneflow_max_size:
                    torch.manual_seed(0)
                    _last_points = last_points[torch.randperm(last_points.shape[0])[:args.sceneflow_max_size]]
                break
        if first_idx is None or last_idx is None:
            cluster_result['state'] = 'invalid'
            continue

        # Perform sceneflow estimation for first and last frames
        max_inlier_dist = args.sceneflow_max_inlier_dist
        T_hist, T_icp3dof = get_histogram_based_and_icp_based_transformations(_first_points.clone().cpu(), _last_points.clone().cpu(), 
                            args.sceneflow_search_size, args.sceneflow_search_step, args.sceneflow_max_icp_iterations)
        T_hist, T_icp3dof = T_hist.cuda(), T_icp3dof.cuda()
        dist_static = torch.norm(first_points[:, :3].reshape(-1, 1, 3) - last_points[:, :3].reshape(1, -1, 3), dim=2)
        min_dist_static1, min_dist_static2 = torch.min(dist_static, dim=1).values, torch.min(dist_static, dim=0).values
        inlier_ratio_static = ((min_dist_static1 < max_inlier_dist).float().mean() + (min_dist_static2 < max_inlier_dist).float().mean()) / 2
        
        dists_hist = torch.norm((first_points[:, :3] + T_hist[:3, 3]).reshape(-1, 1, 3) - last_points[:, :3].reshape(1, -1, 3), dim=2)
        min_dist_hist1, min_dist_hist2 = torch.min(dists_hist, dim=1).values, torch.min(dists_hist, dim=0).values
        inlier_ratio_hist = ((min_dist_hist1 < max_inlier_dist).float().mean() + (min_dist_hist2 < max_inlier_dist).float().mean()) / 2
        
        dists_icp = torch.norm(((T_icp3dof[:3, :3] @ first_points[:, :3].T).T + T_icp3dof[:3, 3]).reshape(-1, 1, 3) - last_points[:, :3].reshape(1, -1, 3), dim=2)
        min_dist_icp1, min_dist_icp2 = torch.min(dists_icp, dim=1).values, torch.min(dists_icp, dim=0).values
        inlier_ratio_icp = ((min_dist_icp1 < max_inlier_dist).float().mean() + (min_dist_icp2 < max_inlier_dist).float().mean()) / 2

        if inlier_ratio_icp > inlier_ratio_hist:
            inlier_ratio_dynamic = inlier_ratio_icp
            T_sceneflow = T_icp3dof
        else:
            inlier_ratio_dynamic = inlier_ratio_hist
            T_sceneflow = T_hist
        if inlier_ratio_dynamic < inlier_ratio_static:
            cluster_result['state'] = 'static'
            cluster_result['inlier_ratio'] = inlier_ratio_static.item()
        else:
            cluster_result['state'] = 'dynamic'
            cluster_result['inlier_ratio'] = inlier_ratio_dynamic.item()

            # Fit new bounding box for dynamic cluster
            t_first = lidar_with_sweeps[first_idx]['timestamp']
            t_last = lidar_with_sweeps[last_idx]['timestamp']
            step_first2last = round((t_last - t_first) * args.lidar_freq)
            T_onestep = torch.from_numpy(scipy.linalg.expm(scipy.linalg.logm(T_sceneflow.cpu(), disp=False)[0]/(step_first2last))).real.float().cuda()
            T_onestep_inv = torch.linalg.inv(T_onestep)
            cluster_points_fine = []
            t_sample = lidar_with_sweeps[sample_idx]['timestamp']
            for i in range(first_idx, last_idx):
                t_i = lidar_with_sweeps[i]['timestamp']
                step_i2sample = round((t_sample - t_i) * args.lidar_freq)
                cluster_points_i = cluster_points[cluster_points[:, 3] == i]
                if step_i2sample == 0:
                    cluster_points_fine.append(cluster_points_i[:, :3])
                elif step_i2sample > 0:
                    T_i2sample = torch.linalg.matrix_power(T_onestep, step_i2sample)
                    cluster_points_fine.append((T_i2sample[:3, :3] @ cluster_points_i[:, :3].T).T + T_i2sample[:3, 3])
                else:
                    T_i2sample = torch.linalg.matrix_power(T_onestep_inv, -step_i2sample)
                    cluster_points_fine.append((T_i2sample[:3, :3] @ cluster_points_i[:, :3].T).T + T_i2sample[:3, 3])
            cluster_points_fine = torch.cat(cluster_points_fine, dim=0)
            T_reference_bbox, bboxdimensions = fit_bounding_box(cluster_points_fine)
            x, y, z = T_reference_bbox[:3, 3].tolist()
            l, w, h = bboxdimensions
            yaw = torch.atan2(T_reference_bbox[1, 0], T_reference_bbox[0, 0]).item()

            # 取bbox附近地面点的平均高度作为bbox的底部高度
            R_inv = T_reference_bbox[:3, :3].T
            T_inv = -R_inv @ T_reference_bbox[:3, 3]
            ground_points_r = (R_inv @ ground_points_c[:, :3].T).T + T_inv
            ground_points_r = ground_points_r[(abs(ground_points_r[:, 0] - x) < l / 2) &
                                            (abs(ground_points_r[:, 1] - y) < w / 2)]
            height_above_ground = 0
            if ground_points_r.shape[0] > 10:
                height_above_ground = ground_points_r[:, 2].mean()
                z = z - height_above_ground
                h = h + height_above_ground
                
            vx, vy, vz = (T_sceneflow[:3, 3] * args.lidar_freq).tolist()
            cluster_result['bbox_3d_fine'] = [x, y, z, l, w, h, yaw, vx, vy, vz]

    print(f"Point cloud loading: {time2 - time1:.2f}s, "
          f"Clustering: {time3 - time2:.2f}s, "
          f"Bounding box fitting`: {time4 - time3:.2f}s, "
          f"Sceneflow estimation: {time.time() - time4:.2f}s, "
          f"Total: {time.time() - time1:.2f}s")
    # Save the clustering results
    lidar_path = lidar_with_sweeps[sample_idx]['lidar_path']
    save_path = os.path.join(scene_save_dir, os.path.basename(lidar_path).replace('.bin', '.pkl'))
    with open(save_path, 'wb') as f:
        pickle.dump({
            'points_c': points_c.cpu().numpy(),
            'ground_points_c': ground_points_c.cpu().numpy(),
            'cluster_results': cluster_results,
        }, f)


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--frame_len', type=int, default=11, help='Number of frames to cluster at once')
    parser.add_argument('--pc_range', type=float, nargs=6, default=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], help='Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]')
    parser.add_argument('--cluster_min_size', type=int, default=16, help='Minimum size of clusters to be considered valid')
    parser.add_argument('--cluster_selection_epsilon', type=float, default=0.5, help='A distance threshold. Clusters below this value will be merged')
    parser.add_argument('--cluster_max_z_above_ground', type=float, default=1.0, help='Maximum height of clusters above ground to be considered valid')
    parser.add_argument('--cluster_size_range', type=float, nargs=6, default=[0.25, 0.25, 0.25, 20, 6, 6], help='Range of cluster size to consider for filtering [l_min, w_min, h_min, l_max, w_max, h_max]')
    parser.add_argument('--cluster_min_area', type=float, default=0.15, help='Minimum area of clusters for filtering')
    parser.add_argument('--cluster_max_ratio', type=float, default=8.0, help='Maximum ratio of length to width for clusters')
    parser.add_argument('--sceneflow_min_size', type=int, default=16, help='Minimum size of single frame clusters for sceneflow estimation')
    parser.add_argument('--sceneflow_max_size', type=int, default=800, help='Maximum size of single frame clusters for sceneflow estimation')
    parser.add_argument('--sceneflow_search_size', type=float, default=10.0, help='Search size in x and y direction for sceneflow estimation')
    parser.add_argument('--sceneflow_search_step', type=float, default=0.1, help='Search step in x and y direction for sceneflow estimation')
    parser.add_argument('--sceneflow_max_icp_iterations', type=int, default=10, help='Maximum number of ICP iterations for sceneflow estimation')
    parser.add_argument('--sceneflow_max_inlier_dist', type=float, default=0.3, help='Maximum inlier distance for sceneflow estimation')
    parser.add_argument('--lidar_freq', type=int, default=20, help='Lidar frequency in Hz')
    parser.add_argument('--info_path', type=str, default='./data/nuscenes/nuscenes_data_info.pkl', help='Path to the NuScenes data info file')
    parser.add_argument('--is_ground_dir', type=str, default='./data/nuscenes/is_ground', help='Directory containing ground removal results')
    parser.add_argument('--save_dir', type=str, default='./data/nuscenes/3d_objects', help='Directory to save the clustering results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)

    for scene in data_info:
        scene_name = scene['scene_name']
        scene_save_dir = f"{args.save_dir}/{scene_name}"
        os.makedirs(scene_save_dir, exist_ok=True)

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
            })
            sample_idx_list.append(count)
            count += 1

        with torch.no_grad():
            for sample_idx in sample_idx_list:
                cluster_and_sceneflow(lidar_with_sweeps, sample_idx, scene_save_dir)

        print(f"Processed scene: {scene_name}, total samples: {len(sample_idx_list)}")

    print("Clustering and sceneflow estimation completed for all scenes.")