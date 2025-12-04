import os
import argparse
import numpy as np
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--info_path', type=str, default='./data/nuscenes/nuscenes_data_info.pkl', help='Path to the nuscenes data info file')
    parser.add_argument('--pseudo_path', type=str, default='./data/nuscenes/pseudo_labels', help='Path to the pseudo labels directory')
    parser.add_argument('--train_info_path', type=str, default='./pcdet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl', help='Path to the train info file to be loaded')
    parser.add_argument('--save_path', type=str, default='./pcdet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train_pseudo.pkl', help='Path to save the updated train info file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.info_path, 'rb') as f:
        data_info = pickle.load(f)

    with open(args.train_info_path, 'rb') as f:
        train_info = pickle.load(f)

    anchor_sizes = {
        'Vehicle': [3.9, 1.6, 1.56],
        'Pedestrian': [0.8, 0.6, 1.73],
        'Cyclist': [1.76, 0.6, 1.73]
    }

    token2idx = {info['token']: idx for idx, info in enumerate(train_info)}

    count = 0
    for scene in data_info:
        scene_name = scene['scene_name']
        pseudo_labels_path = os.path.join(args.pseudo_path, f'{scene_name}.pkl')
        with open(pseudo_labels_path, 'rb') as f:
            pseudo_labels = pickle.load(f)
        
        for i in range(len(scene['samples'])):
            sample_token = scene['samples'][i]['token']
            if token2idx.get(sample_token) is None:
                continue
            sample_idx = token2idx[sample_token]
            obj_list = pseudo_labels[i]['pseudo_labels_3d']
        
            if len(obj_list) > 0:
                annotations = {}
                annotations['name'] = np.array([obj['name'] for obj in obj_list])
                annotations['center_3d'] = np.concatenate([obj['center_3d'].reshape(1, 3) for obj in obj_list], axis=0)
                annotations['bbox_3d'] = np.concatenate([obj['bbox_3d'].reshape(1, 7) if (obj['bbox_3d'] is not None and obj['score_3d'] > 0.2)
                    else np.array([[obj['center_3d'][0], obj['center_3d'][1], obj['center_3d'][2], 0, 0, 0, 0]]) for obj in obj_list], axis=0)
                annotations['bbox_3d'][:, 2] += annotations['bbox_3d'][:, 5] / 2
                annotations['score'] = np.array([obj['score_3d'] if obj['score_3d'] is not None else obj['score_2d'] for obj in obj_list])
                annotations['dynamic'] = np.array([sum(obj['vel_3d'] ** 2) > 1 if obj['bbox_3d'] is not None else False for obj in obj_list], dtype=bool)
                annotations['ref_size'] = np.array([obj['max_lwh'] if obj['max_lwh'] is not None else anchor_sizes[obj['name']] for obj in obj_list])
                annotations['vel_3d'] = np.concatenate([obj_list[j]['vel_3d'].reshape(1, 3) if annotations['dynamic'][j] else np.array([[0.0, 0.0]]) for j in range(len(obj_list))], axis=0)
                train_info[sample_idx]['pseudo_annos'] = annotations

            count += 1
            if count % 100 == 0:
                print(f'Processed {count} / {len(train_info)} samples')
    
    print(f'Processed {count} / {len(train_info)} samples')

    with open(args.save_path, 'wb') as f:
        pickle.dump(train_info, f)