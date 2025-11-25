import os
import argparse
import numpy as np
import pickle
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

    for sample_idx in range(len(data_info)):
        pseudo_annos = data_info[sample_idx]['pseudo_annos']
        mask = (pseudo_annos['score'] > 0.6) & pseudo_annos['dynamic']
        if mask.sum() > 0:
            anno = {
                'name': pseudo_annos['name'][mask],
                'gt_boxes_lidar': pseudo_annos['bbox_3d'][mask],
                'difficulty': np.zeros(mask.sum(), dtype=np.int32),
                'bbox': np.zeros((mask.sum(), 4), dtype=np.float32)
            }
            for key in pseudo_annos:
                pseudo_annos[key] = pseudo_annos[key][~mask]
            data_info[sample_idx]['annos'] = anno
            data_info[sample_idx]['pseudo_annos'] = pseudo_annos
        else:
            anno = {
                'name': np.array([]),
                'gt_boxes_lidar': np.zeros((0, 7), dtype=np.float32),
                'difficulty': np.zeros((0,), dtype=np.int32),
                'bbox': np.zeros((0, 4), dtype=np.float32)
            }
            data_info[sample_idx]['annos'] = anno

        if (sample_idx + 1) % 100 == 0:
            print(f'Processed {sample_idx + 1} / {len(data_info)} samples')

    with open(args.new_info_path, 'wb') as f:
        pickle.dump(data_info, f)
    print(f'Saved new info to {args.new_info_path}')

    dataset = KittiDataset_pseudo(
        dataset_cfg = args.dataset_cfg,
        class_names = ['Car', 'Pedestrian', 'Cyclist'],
        root_path = args.root_path,
        training=False
    )
    dataset.create_groundtruth_database(info_path=args.new_info_path, split='train_unsupervised')