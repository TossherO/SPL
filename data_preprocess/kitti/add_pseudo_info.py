import os
import argparse
import numpy as np
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--info_path', type=str, default='./data/kitti/kitti_data_info.pkl', help='Path to the KITTI data info file')
    parser.add_argument('--pseudo_path', type=str, default='./data/kitti/pseudo_labels', help='Path to the pseudo labels directory')
    parser.add_argument('--train_info_path', type=str, default='./pcdet/data/kitti/kitti_infos_train_sparse.pkl', help='Path to the train info file to be loaded')
    parser.add_argument('--save_path', type=str, default='./pcdet/data/kitti/kitti_infos_train_sparse_pseudo.pkl', help='Path to save the updated train info file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)

    with open(args.train_info_path, 'rb') as f:
        train_info = pickle.load(f)

    anchor_sizes = {
        'Car': np.array([3.9, 1.6, 1.56]),
        'Pedestrian': np.array([0.8, 0.6, 1.73]),
        'Cyclist': np.array([1.76, 0.6, 1.73])
    }

    for sample_idx in range(len(train_info)):
        sample = data_info['kitti_object'][sample_idx]
        pseudo_labels_path = os.path.join(args.pseudo_path, f'{sample["scene"]}.pkl')
        with open(pseudo_labels_path, 'rb') as f:
            pseudo_labels = pickle.load(f)
        obj_list = pseudo_labels[str(sample['sample_idx']).zfill(10)]['pseudo_labels_3d']
        
        if len(obj_list) > 0:
            annotations = {}
            annotations['name'] = np.array([obj['name'] for obj in obj_list])
            annotations['center_3d'] = np.concatenate([obj['center_3d'].reshape(1, 3) for obj in obj_list], axis=0)
            annotations['bbox_3d'] = np.concatenate([obj['bbox_3d'].reshape(1, 7) if (obj['bbox_3d'] is not None and obj['score_3d'] > 0.2)
                else np.array([[obj['center_3d'][0], obj['center_3d'][1], obj['center_3d'][2], 0, 0, 0, 0]]) for obj in obj_list], axis=0)
            annotations['bbox_3d'][:, 2] += annotations['bbox_3d'][:, 5] / 2
            annotations['score'] = np.array([obj['score_3d'] if obj['score_3d'] is not None else obj['score_2d'] for obj in obj_list])
            annotations['dynamic'] = np.array([sum(obj['vel_3d'] ** 2) > 0 if obj['bbox_3d'] is not None else False for obj in obj_list], dtype=bool)
            annotations['ref_size'] = np.concatenate([obj['max_lwh'] if obj['max_lwh'] is not None else anchor_sizes[obj['name']].reshape(1, 3) for obj in obj_list], axis=0)
            train_info[sample_idx]['pseudo_annos'] = annotations

        if (sample_idx + 1) % 100 == 0:
            print(f'Processed {sample_idx + 1} / {len(train_info)} samples')

    with open(args.save_path, 'wb') as f:
        pickle.dump(train_info, f)