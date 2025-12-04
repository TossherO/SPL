import os
import argparse
import numpy as np
import pickle
import yaml
from easydict import EasyDict
from pathlib import Path
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--root_path', type=str, default='../data/nuscenes/', help='root path of dataset')
    parser.add_argument('--dataset_cfg', type=str, default='cfgs/dataset_configs/nuscenes_dataset_sparse.yaml', help='dataset config file')
    parser.add_argument('--ori_info_path', type=str, default='../data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', help='original info pkl file path')
    parser.add_argument('--new_info_path', type=str, default='../data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train_sparse.pkl', help='new info pkl file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.ori_info_path, 'rb') as f:
        data_info = pickle.load(f)

    counts = {'car': [0, 0], 'truck': [0, 0], 'construction_vehicle': [0, 0], 'bus': [0, 0], 'trailer': [0, 0],
              'barrier': [0, 0], 'motorcycle': [0, 0], 'bicycle': [0, 0], 'pedestrian': [0, 0], 'traffic_cone': [0, 0]}

    for sample_idx in range(len(data_info)):
        gt_names = data_info[sample_idx]['gt_names']
        all_drop_indices = []
        for name in counts:
            num_objects = np.sum(gt_names == name)
            counts[name][1] += num_objects
            if num_objects > 1:
                drop_indices = np.random.choice(np.where(gt_names == name)[0], size=(num_objects - 1), replace=False)
                all_drop_indices.append(drop_indices)
            if num_objects > 0:
                counts[name][0] += 1
        if len(all_drop_indices) > 0:
            all_drop_indices = np.concatenate(all_drop_indices, axis=0)
            keep_indices = np.array([i for i in range(len(gt_names)) if i not in all_drop_indices])
            data_info[sample_idx]['gt_names'] = data_info[sample_idx]['gt_names'][keep_indices]
            data_info[sample_idx]['gt_boxes'] = data_info[sample_idx]['gt_boxes'][keep_indices]
            data_info[sample_idx]['gt_boxes_velocity'] = data_info[sample_idx]['gt_boxes_velocity'][keep_indices]
            data_info[sample_idx]['gt_boxes_token'] = data_info[sample_idx]['gt_boxes_token'][keep_indices]
            data_info[sample_idx]['num_lidar_pts'] = data_info[sample_idx]['num_lidar_pts'][keep_indices]
            data_info[sample_idx]['num_radar_pts'] = data_info[sample_idx]['num_radar_pts'][keep_indices]

        if (sample_idx + 1) % 100 == 0:
            print(f'Processed {sample_idx + 1} / {len(data_info)} samples')

    print('Counts (selected / total):')
    for name in counts:
        print(f'{name}: {counts[name][0]} / {counts[name][1]}')

    with open(args.new_info_path, 'wb') as f:
        pickle.dump(data_info, f)
    print(f'Saved new info to {args.new_info_path}')

    dataset_cfg = EasyDict(yaml.safe_load(open(args.dataset_cfg)))
    dataset = NuScenesDataset(
        dataset_cfg = dataset_cfg,
        class_names = None,
        root_path = Path(args.root_path),
        training = True,
        logger=common_utils.create_logger()
    )
    dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)