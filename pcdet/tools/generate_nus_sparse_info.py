import os
import argparse
import numpy as np
import pickle
import yaml
from easydict import EasyDict
from pathlib import Path
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--root_path', type=str, default='../data/nuscenes/', help='root path of dataset')
    parser.add_argument('--dataset_cfg', type=str, default='cfgs/dataset_configs/nuscenes_dataset_sparse.yaml', help='dataset config file')
    parser.add_argument('--ori_info_path', type=str, default='../data/nuscenes/nuscenes_infos_10sweeps_train.pkl', help='original info pkl file path')
    parser.add_argument('--new_info_path', type=str, default='../data/nuscenes/nuscenes_infos_10sweeps_train_sparse.pkl', help='new info pkl file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.ori_info_path, 'rb') as f:
        data_info = pickle.load(f)

    counts = {'car': [0, 0], 'truck': [0, 0], 'construction_vehicle': [0, 0], 'bus': [0, 0], 'trailer': [0, 0],
              'barrier': [0, 0], 'motorcycle': [0, 0], 'bicycle': [0, 0], 'pedestrian': [0, 0], 'traffic_cone': [0, 0]}

    for sample_idx in range(len(data_info)):
        annos = data_info[sample_idx]['annos']
        for name in counts:
            num_objects = np.sum(annos['name'] == name)
            counts[name][1] += num_objects
            if num_objects > 1:
                drop_indices = np.random.choice(np.where(annos['name'] == name)[0], size=(num_objects - 1), replace=False)
                keep_indices = [i for i in range(len(annos['name'])) if i not in drop_indices]
                for key in annos:
                    annos[key] = annos[key][keep_indices]
            if num_objects > 0:
                counts[name][0] += 1
        data_info[sample_idx]['annos'] = annos

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
        class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                       'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        root_path = Path(args.root_path),
        training = False
    )
    dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)