import os
import argparse
import pickle
import numpy as np
import torch
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from collections import deque


def bbox_iou_2d(bbox1, bbox2):
    '''
    Args:
        bbox1, bbox2: [x, y, w, h] format, where (x, y) is the center of the bounding box,
                      w is the width and h is the height.
    '''
    x1 = bbox1[0] - bbox1[2] / 2
    y1 = bbox1[1] - bbox1[3] / 2
    x2 = bbox1[0] + bbox1[2] / 2
    y2 = bbox1[1] + bbox1[3] / 2

    x1_ = bbox2[0] - bbox2[2] / 2
    y1_ = bbox2[1] - bbox2[3] / 2
    x2_ = bbox2[0] + bbox2[2] / 2
    y2_ = bbox2[1] + bbox2[3] / 2

    inter_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(0, min(y2, y2_) - max(y1, y1_))
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_ - x1_) * (y2_ - y1_)

    return inter_area / (bbox1_area + bbox2_area - inter_area + 1e-6)


def get_best_cluster(points, cluster_indices, img_mask, lidar2img):
    '''
    Get the best cluster from the point cloud data based on the cluster indices and image mask.
    
    Args:
        points (torch.Tensor): Point cloud data (N, 4).
        cluster_indices (torch.Tensor): Cluster indices for each point.
        img_mask (torch.Tensor): Image mask indicating valid points in the image.
        lidar2img (torch.Tensor): Transformation matrix from LiDAR to image coordinates.

    Returns:
        torch.Tensor: Points in the best cluster.
        float: Fit score of the best cluster.
    '''
    cluster_num = cluster_indices.max() + 1
    best_fit_score = 0
    best_cluster_idx = -1
    best_num_points = 0
    max_num_points = 0

    for i in range(cluster_num):
        cluster_points = points[cluster_indices == i, :4].clone()
        if cluster_points.shape[0] == 0:
            continue
        cluster_points[:, 3] = 1
        cluster_points_img = (lidar2img[None, :, :] @ cluster_points[:, :, None]).squeeze(-1)
        cluster_points_img = (cluster_points_img[:, :2] / torch.clamp(cluster_points_img[:, 2:3], min=1e-4))
        in_image = (cluster_points_img[:, 0] >= 0) & (cluster_points_img[:, 0] < img_mask.shape[1]) & \
                    (cluster_points_img[:, 1] >= 0) & (cluster_points_img[:, 1] < img_mask.shape[0])
        cluster_points_img = cluster_points_img[in_image].to(int)
        in_mask = img_mask[cluster_points_img[:, 1], cluster_points_img[:, 0]]
        fit_score = in_mask.sum().item() / cluster_points.shape[0]
        max_num_points = max(max_num_points, cluster_points.shape[0])
        if fit_score + cluster_points.shape[0] / max_num_points > best_fit_score + best_num_points / max_num_points:
            best_fit_score = fit_score
            best_cluster_idx = i
            best_num_points = cluster_points.shape[0]

    if best_cluster_idx == -1:
        return None, 0

    return points[cluster_indices == best_cluster_idx], best_fit_score


def find_neighbors(points_kdtree, query_indices, radius):
    """
    Find neighbors of query points within a given radius from the points.
    Args:
        points_kdtree (KDTree): KDTree built from the points.
        query_indices (np.ndarray): Indices of the query points.
        radius (float): Search radius.
    Returns:
        np.ndarray: Indices of neighbors
    """
    is_neighbor = np.zeros(len(points_kdtree.data), dtype=bool)
    is_neighbor[query_indices] = True
    visited = is_neighbor.copy()
    queue = deque(query_indices)
    query_points = points_kdtree.data[query_indices]

    while queue:
        cur_idx = queue.popleft()
        neighbors = points_kdtree.query_ball_point(points_kdtree.data[cur_idx], radius)
        for n_idx in neighbors:
            if visited[n_idx]:
                continue
            visited[n_idx] = True
            if np.linalg.norm(points_kdtree.data[n_idx][None, :3] - query_points[:, :3], axis=1).min() < 4 * radius:
                is_neighbor[n_idx] = True
                queue.append(n_idx)

    return np.where(is_neighbor)[0]


def remove_outliers(points, std_ratio=2.5):
    """
    Remove outliers from the point cloud based on statistical analysis.
    Args:
        points (torch.Tensor): Point cloud data (N, 3).
        std_ratio (float): Standard deviation ratio for outlier detection.
    Returns:
        torch.Tensor: Filtered point cloud indices.
    """
    dist = torch.norm(points[:, None, :] - points[None, :, :], dim=-1)  # Pairwise distances
    dist_min10 = torch.topk(dist, k=10, dim=1, sorted=False).values[:, 1:].mean(dim=1)  # Mean of the 10 nearest neighbors (excluding self)
    mean_dist = dist_min10.mean()
    std_dist = dist_min10.std()
    threshold = mean_dist + std_ratio * std_dist
    return torch.where(dist_min10 < threshold)[0]  # Return indices of points that are not outliers


def knn_majority_voting(points_kdtree, targets):
    """    
    Perform KNN majority voting to assign target IDs to points.
    Args:
        points_kdtree (KDTree): KDTree built from the points.
        targets (list): List of target point indices, where each index corresponds to a target point set.
    Returns:
        np.ndarray: Array of assigned target IDs for each point.
    """
    N = len(points_kdtree.data)
    # Step 1: 标记可靠点(R)和重叠点(O)
    assigned_target = np.full(N, -1, dtype=int)  # 初始化为-1，表示未分配
    assigned_counts = np.zeros(N, dtype=int)  # 目标点集计数
    # 高效分配目标ID和重叠掩码
    for idx, t in enumerate(targets):
        assigned_target[t] = idx  # 直接批量分配
        assigned_counts[t] += 1  # 统计每个目标点集的点数
    
    k = 10  # 近邻数，可调整
    # 预计算目标中心（用于后备）
    target_centers = []
    for t in targets:
        if len(t) == 0:
            target_centers.append(np.zeros(3))  # 如果目标点集为空，使用零向量
        else:
            target_points = points_kdtree.data[t]  # 获取目标点坐标
            center = np.mean(target_points, axis=0)
            target_centers.append(center)
    
    # Step 2: 处理重叠点
    for i in np.where(assigned_counts > 1)[0]:  # 对于每个重叠点集
        # 查找k近邻索引
        distances, indices = points_kdtree.query([points_kdtree.data[i]], k=k+1)  # +1排除自身
        neighbors = indices[0][1:]  # 排除第一个（自身）
        
        # 只考虑可靠邻居（非重叠）
        reliable_neighbors = [j for j in neighbors if assigned_counts[j] == 1]

        if reliable_neighbors:
            # 统计可靠邻居的目标频次
            neighbor_targets = [assigned_target[j] for j in reliable_neighbors]
            unique_targets, counts = np.unique(neighbor_targets, return_counts=True)
            best_target = unique_targets[np.argmax(counts)]
        else:
            # 后备策略：计算到各目标中心的距离
            dist_to_centers = [np.linalg.norm(points_kdtree.data[i] - center) for center in target_centers]
            best_target = np.argmin(dist_to_centers)
        assigned_target[i] = best_target
    
    return assigned_target  # 返回每个点的最终目标ID


def get_object_points(frame_info, save_path):
    # 1. Load the point cloud data
    points = np.fromfile(frame_info['lidar_path'], dtype=np.float32).reshape(-1, 4)
    points = torch.tensor(points, dtype=torch.float32).cuda()
    is_ground = np.fromfile(os.path.join(args.is_ground_dir, frame_info['scene'], frame_info['name'] + '.bin'), dtype=bool)
    is_ground = torch.tensor(is_ground, dtype=torch.bool).cuda()
    points = points[~is_ground]

    # 2. Filter points based on the point cloud range
    pc_range = torch.tensor(np.array(args.pc_range, dtype=np.float32).reshape(2, 3)).cuda()
    mask = torch.all((points[:, :3] >= pc_range[0]) & (points[:, :3] <= pc_range[1]), axis=1)
    points = points[mask]

    # 3. Load image processing results
    with open(frame_info['img_result_path'], 'rb') as f:
        img_result = pickle.load(f)
    cam2img = np.eye(4)
    cam2img[:3, :] = frame_info['cam2img']
    lidar2cam = frame_info['lidar2cam']
    intrinsic = cam2img[:3, :3]
    lidar2img = torch.tensor(cam2img @ lidar2cam, dtype=torch.float32).cuda()

    # 4. Project points to image
    points_homo = points.clone()
    points_homo[:, 3] = 1  # Set homogeneous coordinate
    points_img = (lidar2img @ points_homo.T).T
    points_img[:, :2] = points_img[:, :2] / torch.clamp(points_img[:, 2:3], min=1e-4)
    points_img[:, 3] = torch.arange(points_homo.shape[0], dtype=torch.float32).cuda()
    points_img_mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < args.img_hw[1]) & \
                      (points_img[:, 1] >= 0) & (points_img[:, 1] < args.img_hw[0])
    
    # 5. Get the corresponding points for each object
    _points = points[:, :3].clone()
    _points[:, 2] = _points[:, 2] / 4
    points_sample_kdtree = KDTree(_points.cpu().numpy())
    objects = []
    points_temp = points_img[points_img_mask].to(int)  # [N, 3]
    bboxes = img_result['bboxes']
    labels = img_result['labels']
    ids = img_result['ids']
    object_contours = img_result['masks']
    scores = img_result['scores']
    names = [args.img_labels[label] for label in labels]
    points_object_2d = []

    # 5.1 Get the points for each object by object segmentation mask
    for i in range(len(labels)):
        img_mask = np.zeros(args.img_hw, dtype=np.uint8)
        cv2.drawContours(img_mask, [object_contours[i]], -1, 255, thickness=cv2.FILLED)
        kernel = np.ones((3, 3), dtype=np.uint8)
        area = np.sum(img_mask > 0)
        iterations = np.ceil(np.sqrt(area) // 20).astype(int)
        img_mask = cv2.dilate(img_mask, kernel, iterations=iterations)
        img_mask = torch.tensor(img_mask.astype(bool)).cuda()
        # 取出mask中为True的点
        points_temp_masked = points_temp[img_mask[points_temp[:, 1], points_temp[:, 0]]]
        points_object_2d.append(points_temp_masked)

    # 5.2 Handle riding objects (bicycle, motorcycle)
    riding_idx = [i for i in range(len(labels)) if names[i] in ['bicycle', 'motorcycle']]
    person_riding_pairs = {}
    for i in riding_idx:
        bbox = bboxes[i]
        person_idx = [k for k in range(len(labels)) if names[k] == 'person']
        # 计算骑行物体与所有人的IOU，如果存在某个人与之相交，则视其为合法目标
        best_iou, best_idx = 0, -1
        for k in person_idx:
            iou = bbox_iou_2d(bbox, bboxes[k])
            if iou > args.riding_iou_threshold and iou > best_iou and bboxes[k][1] < bbox[1]:
                best_iou, best_idx = iou, k
        if best_idx != -1:
            names[i] = names[i] + '_used'
            names[best_idx] = names[best_idx] + '_rider'
            points_object_2d[i] = torch.cat([points_object_2d[i], points_object_2d[best_idx]], dim=0)
            points_object_2d[best_idx] = torch.zeros((0, 4), dtype=torch.float32).cuda()
            person_riding_pairs[i] = best_idx
            
    # 5.3 Filter points based on object height range
    for i in range(len(labels)):
        if points_object_2d[i].shape[0] == 0:
            continue
        height_range = args.objects_height_range[names[i]]
        if names[i] in ['bicycle_used', 'motorcycle_used']:
            rider_contour = object_contours[person_riding_pairs[i]]
            riding_contour = object_contours[i]
            height_img = max(rider_contour[:, 1].max() - rider_contour[:, 1].min(),
                                riding_contour[:, 1].max() - riding_contour[:, 1].min())
        else:
            height_img = object_contours[i][:, 1].max() - object_contours[i][:, 1].min()
        # 根据相机内参和物体高度范围计算对应的点云深度范围: z = fy * h / h_img
        d_min = intrinsic[1, 1] * height_range[0] / height_img
        d_max = intrinsic[1, 1] * height_range[1] / height_img
        # 过滤点云中对应的点
        depth_mask = (points_object_2d[i][:, 2] >= d_min) & (points_object_2d[i][:, 2] <= d_max)
        points_object_2d[i] = points_object_2d[i][depth_mask]

    # 5.4 Adjust points in 3d space
    for i in range(len(labels)):
        name = names[i].split('_')[0]
        if points_object_2d[i].shape[0] < args.cluster_min_size:
            points_object_indices = points_object_2d[i][:, 3].long().cpu().numpy() if points_object_2d[i].shape[0] > 0 else None
        else:
            # Filter noise points by DBSCAN clustering
            img_mask = np.zeros(args.img_hw, dtype=np.uint8)
            if names[i] in ['bicycle_used', 'motorcycle_used']:
                contours = [object_contours[i], object_contours[person_riding_pairs[i]]]
            else:
                contours = [object_contours[i]]
            cv2.drawContours(img_mask, contours, -1, 255, thickness=cv2.FILLED)
            img_mask = torch.tensor(img_mask.astype(bool)).cuda()
            points_object_3d = points[points_object_2d[i][:, 3].long()].clone()
            points_object_3d[:, 3] = points_object_2d[i][:, 3]
            _points_object_3d = points_object_3d[:, :3].clone()
            _points_object_3d[:, 2] = _points_object_3d[:, 2] / 4
            dbscan_clusterer = DBSCAN(min_samples=args.cluster_min_size, eps=args.neighbor_radius[name])
            cluster_indices = dbscan_clusterer.fit_predict(_points_object_3d.cpu().numpy())
            cluster_indices = torch.tensor(cluster_indices, dtype=torch.int32).cuda()
            cluster_points, fit_score = get_best_cluster(points_object_3d, cluster_indices, img_mask, lidar2img)

            # Get more neighbors of the points
            if cluster_points is None or fit_score < 0.5:
                points_object_indices = points_object_2d[i][:, 3].long().cpu().numpy() if points_object_2d[i].shape[0] > 0 else None
            else:
                points_object_indices = find_neighbors(points_sample_kdtree, cluster_points[:, 3].long().cpu().numpy(), args.neighbor_radius[name] / 2)

            # Filter outliers based on statistical analysis
            # if name in ['person'] and points_object_indices is not None and points_object_indices.shape[0] > 10:
            #     points_object_3d = points_sample[points_object_indices].clone()  # [N, 5]
            #     _points_object_3d = points_object_3d[:, :3].clone()
            #     _points_object_3d[:, 2] = _points_object_3d[:, 2] / 4
            #     points_object_indices = points_object_indices[remove_outliers(_points_object_3d).cpu().numpy()]

        # Save the points for the object
        obj = {
            'name': names[i],
            'id': ids[i] if ids is not None else -1,
            'bbox_2d': bboxes[i],
            'mask_2d': object_contours[i],
            'score_2d': scores[i],
            'points': points_object_indices if points_object_indices is not None else None
        }
        if names[i] in ['bicycle_used', 'motorcycle_used']:
            obj['rider'] = {
                'id': ids[person_riding_pairs[i]] if ids is not None else -1,
                'bbox_2d': bboxes[person_riding_pairs[i]],
                'mask_2d': object_contours[person_riding_pairs[i]],
                'score_2d': scores[person_riding_pairs[i]]
            }
        objects.append(obj)

    # 6. Process the overlapping objects
    points_object_indices_list = []
    for obj in objects:
        if obj['points'] is not None:
            points_object_indices_list.append(obj['points'])
        else:
            points_object_indices_list.append(np.array([], dtype=np.int32))
    assigned_target = knn_majority_voting(points_sample_kdtree, points_object_indices_list)
    for i, obj in enumerate(objects):
        if obj['points'] is not None:
            obj['points'] = np.where(assigned_target == i)[0]
            if obj['points'].shape[0] == 0:
                obj['points'] = None

    # 7. Merge the same vehicle objects
    # vehicle_idx = [i for i, obj in enumerate(objects) if obj['name'] in ['car', 'truck', 'bus'] and obj['points'] is not None]
    # for i in range(len(vehicle_idx)):
    #     for j in range(i + 1, len(vehicle_idx)):
    #         if objects[vehicle_idx[i]]['cam'] == objects[vehicle_idx[j]]['cam']:
    #             continue
    #         points1 = points_sample[objects[vehicle_idx[i]]['points']]
    #         points2 = points_sample[objects[vehicle_idx[j]]['points']]
    #         dist = torch.norm(points1[:, None, :3] - points2[None, :, :3], dim=2)
    #         min_dist1, min_dist2 = torch.min(dist, dim=1).values, torch.min(dist, dim=0).values
    #         sum1, sum2 = torch.sum(min_dist1 < args.vehicle_merge_distance), torch.sum(min_dist2 < args.vehicle_merge_distance)
    #         if sum1 > args.vehicle_merge_min_points and sum2 > args.vehicle_merge_min_points:
    #             objects[vehicle_idx[j]]['points'] = np.concatenate([objects[vehicle_idx[i]]['points'], objects[vehicle_idx[j]]['points']], axis=0)
    #             objects[vehicle_idx[i]]['points'] = None
    #             objects[vehicle_idx[i]]['merged_idx'] = vehicle_idx[j]
    #             break

    results = {
        'objects': objects,
        'lidar2img': lidar2img.cpu().numpy(),
    }
    # Save the results
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    return


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pc_range', type=float, nargs=6, default=[0, -40.0, -3.0, 70.4, 40.0, 1.0], help='Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]')
    parser.add_argument('--img_hw', type=int, nargs=2, default=[375, 1242], help='Image width and height for projection')
    parser.add_argument('--img_labels', type=dict, default={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}, help='Mapping of label indices to object names')
    parser.add_argument('--riding_iou_threshold', type=float, default=0.2, help='IOU threshold for filtering riding objects')
    parser.add_argument('--objects_height_range', type=dict, default={'person': [1.0, 2.0], 'bicycle': [0.5, 2.0], 'bicycle_used': [1.0, 2.0], 'car': [1.0, 3.0], 'motorcycle': [0.5, 2.0],
                                                                      'motorcycle_used': [1.0, 2.0]}, help='Height range for different object types')
    parser.add_argument('--neighbor_radius', type=dict, default={'person': 0.3, 'bicycle': 0.3, 'car': 0.5, 'motorcycle': 0.3}, help='Radius for finding neighbors in clustering')
    parser.add_argument('--cluster_min_size', type=int, default=4, help='Minimum size of clusters to be considered valid')
    # parser.add_argument('--vehicle_merge_distance', type=float, default=0.4, help='Distance threshold for merging vehicle clusters from different cameras')
    # parser.add_argument('--vehicle_merge_min_points', type=int, default=5, help='Minimum number of points for merging vehicle clusters')
    parser.add_argument('--info_path', type=str, default='./data/kitti/kitti_data_info.pkl', help='Path to the KITTI data info file')
    parser.add_argument('--is_ground_dir', type=str, default='./data/kitti/is_ground', help='Directory containing ground removal results')
    parser.add_argument('--img_results_dir', type=str, default='./data/kitti/img_results', help='Directory containing image processing results')
    parser.add_argument('--save_dir', type=str, default='./data/kitti/3d_objects', help='Directory to save the processed 3D objects')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)

    count = 0
    for sample in data_info['kitti_object']:
        scene = sample['scene']
        scene_save_dir = f"{args.save_dir}/{scene}"
        os.makedirs(scene_save_dir, exist_ok=True)

        start = sample['start_idx']
        end = sample['end_idx']

        for i in range(start, end + 1):
            name = str(i).zfill(10)
            save_path = os.path.join(scene_save_dir, name + '.pkl')
            if not os.path.exists(save_path):
                frame_info = {
                    'scene': scene,
                    'name': name,
                    'lidar_path': data_info['kitti_raw'][scene][name]['lidar_path'],
                    'img_result_path': os.path.join(args.img_results_dir, scene, name + '.pkl'),
                    'cam2img': sample['cam2img'],
                    'lidar2cam': sample['lidar2cam'],
                }
                with torch.no_grad():
                    get_object_points(frame_info, save_path)

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} / {len(data_info['kitti_object'])} samples.")