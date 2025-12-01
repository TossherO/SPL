import os
import argparse
import numpy as np
import pickle
import yaml
from easydict import EasyDict
from pathlib import Path
from pcdet.datasets.kitti.kitti_dataset_pseudo import KittiDataset_pseudo


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--root_path', type=str, default='../data/kitti/', help='root path of dataset')
    parser.add_argument('--dataset_cfg', type=str, default='cfgs/dataset_configs/kitti_dataset_sparse.yaml', help='dataset config file')
    parser.add_argument('--ori_info_path', type=str, default='../data/kitti/kitti_infos_train_sparse_pseudo.pkl', help='original info pkl file path')
    parser.add_argument('--new_info_path', type=str, default='../data/kitti/kitti_infos_train_unsupervised.pkl', help='new info pkl file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.ori_info_path, 'rb') as f:
        data_info = pickle.load(f)

    anchor_sizes = {
        'Car': [3.9, 1.6, 1.56],
        'Pedestrian': [0.8, 0.6, 1.73],
        'Cyclist': [1.76, 0.6, 1.73]
    }
    counts = {'Car': [0, 0], 'Pedestrian': [0, 0], 'Cyclist': [0, 0]}

    for sample_idx in range(len(data_info)):
        pseudo_annos = data_info[sample_idx].get('pseudo_annos', None)
        if pseudo_annos is not None:
            bboxes = pseudo_annos['bbox_3d'].copy()
            mask1 = (pseudo_annos['score'] > 0.4) & (pseudo_annos['dynamic'] | (pseudo_annos['name'] == 'Car'))
            for i in range(len(pseudo_annos['name'])):
                if pseudo_annos['name'][i] == 'Car':
                    bbox = bboxes[i]
                    max_lwh = pseudo_annos['ref_size'][i]
                    if ((bbox[3] > 0.8 * max_lwh[0]) & (bbox[3] < 1.2 * max_lwh[0]) &
                        (bbox[4] > 0.8 * max_lwh[1]) & (bbox[4] < 1.2 * max_lwh[1]) &
                        (bbox[5] > 0.8 * max_lwh[2]) & (bbox[5] < 1.2 * max_lwh[2])):
                        bboxes[i][3:6] = max_lwh
                    else:
                        bboxes[i][3:6] = 0
            anchors = np.array([anchor_sizes[name] for name in pseudo_annos['name']])
            min_size = anchors * 0.7
            max_size = anchors * 1.3
            mask2 = (bboxes[:, 3] > min_size[:, 0]) & (bboxes[:, 3] < max_size[:, 0]) & \
                    (bboxes[:, 4] > min_size[:, 1]) & (bboxes[:, 4] < max_size[:, 1]) & \
                    (bboxes[:, 5] > min_size[:, 2]) & (bboxes[:, 5] < max_size[:, 2])
            mask = mask1 & mask2
            bboxes = bboxes[mask]
            for i in range(len(bboxes)):
                if pseudo_annos['name'][mask][i] == 'Car':
                    continue
                for j in range(3):
                    if bboxes[i][3 + j] < anchors[mask][i][j]:
                        x1, x2 = bboxes[i][3 + j], anchors[mask][i][j]
                        bboxes[i][3 + j] = (x1 * x1 + x2 * x2) / (x1 + x2)

        if pseudo_annos is not None and mask.sum() > 0:
            annos = {
                'name': pseudo_annos['name'][mask],
                'gt_boxes_unsup': bboxes,
                'score': pseudo_annos['score'][mask],
                'difficulty': np.zeros(mask.sum(), dtype=np.int32),
                'bbox': np.zeros((mask.sum(), 4), dtype=np.float32)
            }
            for name in counts:
                counts[name][0] += np.sum(annos['name'] == name)
                counts[name][1] += np.sum(pseudo_annos['name'] == name)
            for key in pseudo_annos:
                pseudo_annos[key] = pseudo_annos[key][~mask]
            data_info[sample_idx]['annos'] = annos
            data_info[sample_idx]['pseudo_annos'] = pseudo_annos
        else:
            annos = {
                'name': np.array([]),
                'gt_boxes_unsup': np.zeros((0, 7), dtype=np.float32),
                'difficulty': np.zeros((0,), dtype=np.int32),
                'bbox': np.zeros((0, 4), dtype=np.float32)
            }
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
    dataset = KittiDataset_pseudo(
        dataset_cfg = dataset_cfg,
        class_names = ['Car', 'Pedestrian', 'Cyclist'],
        root_path = Path(args.root_path),
        training = False
    )
    dataset.create_groundtruth_database(info_path=args.new_info_path, split='train_unsupervised')