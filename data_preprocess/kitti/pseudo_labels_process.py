import os
import argparse
import pickle
import numpy as np
import torch
import cv2
import time


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


def fit_bounding_box_fix_yaw(points, yaw):
    device = points.device

    yaw = torch.tensor(yaw, dtype=torch.float32, device=device)
    R = torch.tensor([
        [torch.cos(yaw), torch.sin(yaw)],
        [-torch.sin(yaw), torch.cos(yaw)]
    ], device=device)

    rotated_points = (R @ points[:, :2].T).T
    min_x = rotated_points[:, 0].min()
    max_x = rotated_points[:, 0].max()
    min_y = rotated_points[:, 1].min()
    max_y = rotated_points[:, 1].max()

    corners_local = torch.tensor([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y]
    ], device=device)
    corners_ref = (R.T @ corners_local.T).T

    bbox_center_ref = torch.zeros(3, device=device)
    bbox_center_ref[0] = corners_ref[:, 0].mean()
    bbox_center_ref[1] = corners_ref[:, 1].mean()
    bbox_center_ref[2] = points[:, 2].min()

    T_matrix = torch.eye(4, device=device)
    T_matrix[:3, 3] = bbox_center_ref
    T_matrix[:2, :2] = R.T

    length = torch.norm(corners_ref[1] - corners_ref[0])
    width = torch.norm(corners_ref[3] - corners_ref[0])
    height = points[:, 2].max() - points[:, 2].min()
    
    return T_matrix, [length.item(), width.item(), height.item()]


def bbox_3d_to_corners(bbox_3d):
    x, y, z, l, w, h, yaw = bbox_3d
    bbox_3d_corners = np.array([
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2]
    ])
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    bbox_3d_corners = (rotation_matrix @ bbox_3d_corners.T).T + np.array([x, y, z + h / 2])
    return bbox_3d_corners


def get_fitting_score(img_mask, bbox_3d_corners, lidar2img):
    corners_homo = np.hstack((bbox_3d_corners, np.ones((bbox_3d_corners.shape[0], 1))))
    corners_img = (lidar2img @ corners_homo.T).T
    corners_img = (corners_img[:, :2] / np.maximum(corners_img[:, 2:3], 1e-4)).astype(np.float32)
    in_img = (corners_img[:, 0] >= 0) & (corners_img[:, 0] < args.img_hw[1]) & (corners_img[:, 1] >= 0) & (corners_img[:, 1] < args.img_hw[0])
    if in_img.sum() == 0:
        return 0.
    hull = cv2.convexHull(corners_img).astype(np.int32)
    mask = np.zeros(args.img_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    inter_area = np.logical_and(mask, img_mask).sum()
    union_area = np.logical_or(mask, img_mask).sum()
    if union_area == 0:
        return 0.
    return inter_area / union_area


def process_3d_objects(objects, points, ground_points, lidar2img):
    for i in range(len(objects)):
        points_object_indices = objects[i]['points']
        objects[i]['center_3d'] = None
        objects[i]['bbox_3d'] = None
        objects[i]['score_3d'] = None

        if points_object_indices is None:
            continue
        points_object = points[points_object_indices]
        objects[i]['points'] = points_object

        if objects[i]['name'] == 'car':
            if points_object.shape[0] > 10:
                objects[i]['center_3d'] = points_object[:, :3].mean(dim=0).cpu().numpy()

            if points_object.shape[0] > 20:
                T_reference_bbox, bboxdimensions = fit_bounding_box(points_object)
                x, y, z = T_reference_bbox[:3, 3].tolist()
                l, w, h = bboxdimensions
                yaw = torch.atan2(T_reference_bbox[1, 0], T_reference_bbox[0, 0]).item()
                R_inv = T_reference_bbox[:3, :3].T
                T_inv = -R_inv @ T_reference_bbox[:3, 3]

                ground_points_r = (R_inv @ ground_points[:, :3].T).T + T_inv
                ground_points_r = ground_points_r[(abs(ground_points_r[:, 0]) < l / 2) & (abs(ground_points_r[:, 1]) < w / 2)]
                if ground_points_r.shape[0] > 10:
                    dh = ground_points_r[:, 2].mean().item()
                    if dh < 0:
                        z = z + dh
                        h = h - dh

                if (l < args.car_size_range[0] or l > args.car_size_range[3] or
                    w < args.car_size_range[1] or w > args.car_size_range[4] or
                    h < args.car_size_range[2] or h > args.car_size_range[5]):
                    continue

                # 计算表面贴近比例（SPR）
                points_object_r = (R_inv @ points_object[:, :3].T).T + T_inv
                x_dist = l / 2 - abs(points_object_r[:, 0])
                y_dist = w / 2 - abs(points_object_r[:, 1])
                z_dist = h / 2 - abs(points_object_r[:, 2])
                min_dist = torch.min(torch.stack([x_dist, y_dist, z_dist], dim=1), dim=1).values
                spr = torch.sum(min_dist < 0.2) / points_object.shape[0]
                if spr < 0.8:
                    continue

                objects[i]['bbox_3d'] = np.array([x, y, z, l, w, h, yaw])

        else:
            if points_object.shape[0] > 5:
                objects[i]['center_3d'] = points_object[:, :3].mean(dim=0).cpu().numpy()

            if points_object.shape[0] > 10:
                T_reference_bbox, bboxdimensions = fit_bounding_box(points_object)
                x, y, z = T_reference_bbox[:3, 3].tolist()
                l, w, h = bboxdimensions
                yaw = torch.atan2(T_reference_bbox[1, 0], T_reference_bbox[0, 0]).item()
                R_inv = T_reference_bbox[:3, :3].T
                T_inv = -R_inv @ T_reference_bbox[:3, 3]

                ground_points_r = (R_inv @ ground_points[:, :3].T).T + T_inv
                ground_points_r = ground_points_r[(ground_points_r[:, 0] ** 2 + ground_points_r[:, 1] ** 2) < 1.0]
                if ground_points_r.shape[0] > 10:
                    dh = ground_points_r[:, 2].mean().item()
                    if dh < 0:
                        z = z + dh
                        h = h - dh

                if (l < args.other_size_range[0] or l > args.other_size_range[3] or
                    w < args.other_size_range[1] or w > args.other_size_range[4] or
                    h < args.other_size_range[2] or h > args.other_size_range[5]):
                    objects[i]['bbox_3d'] = None
                    continue

                objects[i]['bbox_3d'] = np.array([x, y, z, l, w, h, yaw])
            
        if objects[i]['bbox_3d'] is not None:
            object_contours = [objects[i]['mask_2d']]
            if objects[i].get('rider', None) is not None:
                object_contours.append(objects[i]['rider']['mask_2d'])
            img_mask = np.zeros(args.img_hw, dtype=np.uint8)
            cv2.drawContours(img_mask, object_contours, -1, 255, thickness=cv2.FILLED)
            bbox_3d_corners = bbox_3d_to_corners(objects[i]['bbox_3d'])
            score_3d = get_fitting_score(img_mask, bbox_3d_corners, lidar2img)
            objects[i]['score_3d'] = score_3d
                
    return objects


def process_stream(stream_results):
    pseudo_labels = [{'pseudo_labels_2d': [], 'pseudo_labels_3d': []} for _ in range(len(stream_results))]
    instances = {}
    instance_token_list = []

    for i in range(len(stream_results)):
        objects = stream_results[i]['objects']
        for obj in objects:
            if obj['name'] == 'person_rider':
                continue
            if obj.get('rider', None) is not None:
                bbox_2d_1 = obj['bbox_2d']  # xywh
                bbox_2d_2 = obj['rider']['bbox_2d']  # xywh
                x1_1 = bbox_2d_1[0] - bbox_2d_1[2] / 2
                y1_1 = bbox_2d_1[1] - bbox_2d_1[3] / 2
                x2_1 = bbox_2d_1[0] + bbox_2d_1[2] / 2
                y2_1 = bbox_2d_1[1] + bbox_2d_1[3] / 2
                x1_2 = bbox_2d_2[0] - bbox_2d_2[2] / 2
                y1_2 = bbox_2d_2[1] - bbox_2d_2[3] / 2
                x2_2 = bbox_2d_2[0] + bbox_2d_2[2] / 2
                y2_2 = bbox_2d_2[1] + bbox_2d_2[3] / 2
                x1, y1, x2, y2 = min(x1_1, x1_2), min(y1_1, y1_2), max(x2_1, x2_2), max(y2_1, y2_2)
                obj['bbox_2d'] = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, x2 - x1, y2 - y1])
                obj['mask_2d'] = [obj['mask_2d'], obj['rider']['mask_2d']]
                obj['score_2d'] = max(obj['score_2d'], obj['rider']['score_2d'])

            if obj['center_3d'] is None:
                if obj['score_2d'] > 0.2:
                    pseudo_labels[i]['pseudo_labels_2d'].append({
                        'name': obj['name'],
                        'bbox_2d': obj['bbox_2d'],
                        'mask_2d': obj['mask_2d'],
                        'score_2d': obj['score_2d'],
                    })
            else:
                if obj['id'] == -1:
                    instance_token = f"{time.time()}"
                else:
                    instance_token = f"{obj['name'].split('_')[0]}_{obj['id']}"
                if instances.get(instance_token, None) is None:
                    instances[instance_token] = {
                        'name': obj['name'].split('_')[0],
                        'data_list': [],
                    }
                    instance_token_list.append(instance_token)
                instances[instance_token]['data_list'].append({
                    'sample_idx': i,
                    'bbox_2d': obj['bbox_2d'],
                    'mask_2d': obj['mask_2d'],
                    'score_2d': obj['score_2d'],
                    'center_3d': obj['center_3d'],
                    'bbox_3d': obj['bbox_3d'],
                    'score_3d': obj['score_3d'],
                    'points': obj['points']
                })

    class_map = {
        'car': 'Car',
        'motorcycle': 'Cyclist',
        'bicycle': 'Cyclist',
        'person': 'Pedestrian'
    }
    instance_counts = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}

    for instance_token in instance_token_list:
        instance = instances[instance_token]
        label = class_map[instance['name']]

        if len(instance['data_list']) == 1:
            vel_list = [np.array([0, 0, 0])]
        else:
            t_list = []
            rot_list = []
            pos_list = []
            for data in instance['data_list']:
                t_list.append(stream_results[data['sample_idx']]['timestamp'])
                lidar2global = stream_results[data['sample_idx']]['lidar2global']
                R = lidar2global[:3, :3]
                t = lidar2global[:3, 3]
                rot_list.append(R)
                pos_list.append(R @ data['center_3d'] + t)
            d_list = [pos_list[i + 1] - pos_list[i] for i in range(len(pos_list) - 1)]
            dt_list = [t_list[i + 1] - t_list[i] for i in range(len(t_list) - 1)]
            vel_list = []
            vel_list.append(rot_list[0].T @ d_list[0] / dt_list[0])
            for i in range(len(instance['data_list']) - 2):
                v = (d_list[i] / dt_list[i] * dt_list[i + 1] + d_list[i + 1] / dt_list[i + 1] * dt_list[i]) / (dt_list[i] + dt_list[i + 1])
                vel_list.append(rot_list[i + 1].T @ v)
            vel_list.append(rot_list[-1].T @ d_list[-1] / dt_list[-1])

        if label == 'Car':
            max_lwh = [0, 0, 0]
            for i, data in enumerate(instance['data_list']):
                data['vel_3d'] = vel_list[i]
                if data['bbox_3d'] is not None:
                    if sum(vel_list[i] ** 2) > 1:
                        yaw = data['bbox_3d'][6]
                        if np.cos(yaw) * vel_list[i][0] + np.sin(yaw) * vel_list[i][1] < 0:
                            if yaw > 0:
                                data['bbox_3d'][6] = yaw - np.pi
                            else:
                                data['bbox_3d'][6] = yaw + np.pi
                    if data['score_3d'] > 0.4:
                        max_lwh[0] = max(max_lwh[0], data['bbox_3d'][3])
                        max_lwh[1] = max(max_lwh[1], data['bbox_3d'][4])
                        max_lwh[2] = max(max_lwh[2], data['bbox_3d'][5])
            if sum(max_lwh) > 0:
                for data in instance['data_list']:
                    data['max_lwh'] = max_lwh

        elif label == 'Pedestrian':
            for i, data in enumerate(instance['data_list']):
                data['vel_3d'] = vel_list[i]
                if data['bbox_3d'] is not None:
                    if sum(vel_list[i] ** 2) > 1 and data['score_3d'] > 0.4:
                        yaw = np.arctan2(vel_list[i][1], vel_list[i][0])
                        points_object = data['points']
                        T_reference_bbox, bboxdimensions = fit_bounding_box_fix_yaw(points_object, yaw)
                        x, y, z = T_reference_bbox[:3, 3].tolist()
                        l, w, h = bboxdimensions
                        R_inv = T_reference_bbox[:3, :3].T
                        T_inv = -R_inv @ T_reference_bbox[:3, 3]
                        ground_points = stream_results[data['sample_idx']]['ground_points']
                        ground_points_r = (R_inv @ ground_points[:, :3].T).T + T_inv
                        ground_points_r = ground_points_r[(ground_points_r[:, 0] ** 2 + ground_points_r[:, 1] ** 2) < 1.0]
                        if ground_points_r.shape[0] > 10:
                            dh = ground_points_r[:, 2].mean().item()
                            if dh < 0:
                                z = z + dh
                                h = h - dh
                        data['bbox_3d'] = np.array([x, y, z, l, w, h, yaw])
                        object_contours = [data['mask_2d']]
                        img_mask = np.zeros(args.img_hw, dtype=np.uint8)
                        cv2.drawContours(img_mask, object_contours, -1, 255, thickness=cv2.FILLED)
                        bbox_3d_corners = bbox_3d_to_corners(data['bbox_3d'])
                        lidar2img = stream_results[data['sample_idx']]['lidar2img']
                        score_3d = get_fitting_score(img_mask, bbox_3d_corners, lidar2img)
                        data['score_3d'] = score_3d

        else:
            if sum([sum(vel ** 2) for vel in vel_list]) / len(vel_list) > 1:
                for i, data in enumerate(instance['data_list']):
                    data['vel_3d'] = vel_list[i]
                    if data['bbox_3d'] is not None and data['score_3d'] > 0.4:
                        if sum(vel_list[i] ** 2) > 1:
                            yaw = data['bbox_3d'][6]
                            if np.cos(yaw) * vel_list[i][0] + np.sin(yaw) * vel_list[i][1] < 0:
                                if yaw > 0:
                                    data['bbox_3d'][6] = yaw - np.pi
                                else:
                                    data['bbox_3d'][6] = yaw + np.pi
            else:
                continue

        instance_counts[label] += 1
        for data in instance['data_list']:
            pseudo_labels[data['sample_idx']]['pseudo_labels_3d'].append({
                'name': label,
                'id': instance_counts[label],
                'bbox_2d': data['bbox_2d'],
                'mask_2d': data['mask_2d'],
                'score_2d': data['score_2d'],
                'center_3d': data['center_3d'],
                'bbox_3d': data['bbox_3d'],
                'score_3d': data['score_3d'],
                'vel_3d': data['vel_3d'],
                'max_lwh': data.get('max_lwh', None)
            })
    _pseudo_labels = {stream_results[i]['name']: labels for i, labels in enumerate(pseudo_labels)}
    return _pseudo_labels


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pc_range', type=float, nargs=6, default=[0, -40.0, -3.0, 70.4, 40.0, 1.0], help='Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]')
    parser.add_argument('--img_hw', type=int, nargs=2, default=[375, 1242], help='Image width and height for projection')
    parser.add_argument('--car_size_range', type=float, nargs=6, default=[2.0, 1.2, 1.2, 7.0, 2.0, 3.0], help='Range of car sizes [min_length, min_width, min_height, max_length, max_width, max_height]')
    parser.add_argument('--other_size_range', type=float, nargs=6, default=[0.25, 0.2, 0.5, 2.5, 1.0, 2.5], help='Range of other object sizes [min_length, min_width, min_height, max_length, max_width, max_height]')
    parser.add_argument('--info_path', type=str, default='./data/kitti/kitti_data_info.pkl', help='Path to the KITTI data info file')
    parser.add_argument('--is_ground_dir', type=str, default='./data/kitti/is_ground', help='Directory containing ground removal results')
    parser.add_argument('--objects_dir', type=str, default='./data/kitti/3d_objects', help='Directory containing 3D object results')
    parser.add_argument('--save_dir', type=str, default='./data/kitti/pseudo_labels', help='Directory to save the processed pseudo labels')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)

    for scene in data_info['kitti_raw'].keys():
        raw_data = data_info['kitti_raw'][scene]
        frame_names = list(raw_data.keys())
        frame_names.sort()
        scene_pseudo_labels = {}
        stream_results = []
        
        with torch.no_grad():
            for i in range(len(frame_names)):
                name = frame_names[i]
                points = np.fromfile(raw_data[name]['lidar_path'], dtype=np.float32).reshape(-1, 4)
                points = torch.tensor(points, dtype=torch.float32).cuda()
                is_ground = np.fromfile(os.path.join(args.is_ground_dir, scene, name + '.bin'), dtype=bool)
                is_ground = torch.tensor(is_ground, dtype=torch.bool).cuda()
                ground_points = points[is_ground]
                points = points[~is_ground]
                pc_range = torch.tensor(np.array(args.pc_range, dtype=np.float32).reshape(2, 3)).cuda()
                mask = torch.all((points[:, :3] >= pc_range[0]) & (points[:, :3] <= pc_range[1]), axis=1)
                points = points[mask]

                object_path = os.path.join(args.objects_dir, scene, name + '.pkl')
                with open(object_path, 'rb') as f:
                    objects_info = pickle.load(f)
                objects = objects_info['objects']
                lidar2img = objects_info['lidar2img']

                results = process_3d_objects(objects, points, ground_points, lidar2img)
                stream_results.append({
                    'name': name,
                    'timestamp': raw_data[name]['timestamp'],
                    'lidar2global': raw_data[name]['lidar2global'],
                    'lidar2img': lidar2img,
                    'objects': results,
                    'ground_points': ground_points,
                })

                if i == len(frame_names) - 1 or int(frame_names[i+1]) - int(name) > 1:
                    scene_pseudo_labels.update(process_stream(stream_results))
                    stream_results = []

        save_path = f"{args.save_dir}/{scene}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(scene_pseudo_labels, f)

        print(f"Processed scene: {scene} with {len(scene_pseudo_labels)} frames.")
