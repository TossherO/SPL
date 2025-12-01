import os
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils, box_utils
from pcdet.models import load_data_to_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--ori_info_path', type=str, default='../data/kitti/kitti_infos_train_sparse_pseudo.pkl', help='original info pkl file path')
    parser.add_argument('--new_info_path', type=str, default='../data/kitti/kitti_infos_train_sparse_self.pkl', help='new info pkl file path')
    parser.add_argument('--cfg_file', type=str, default='./cfgs/my_models/voxel_rcnn_kitti_prototype.yaml', help='specify the config file')
    parser.add_argument('--ckpt', type=str, default='../output/my_models/voxel_rcnn_kitti_prototype/default/ckpt/checkpoint_epoch_80.pth', help='specify the pretrained model')
    parser.add_argument('--is_unsupervised', action='store_true', help='whether the model is unsupervised')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.ori_info_path, 'rb') as f:
        data_info = pickle.load(f)
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    logger = common_utils.create_logger()

    cfg.DATA_CONFIG.INFO_PATH['test'] = [os.path.basename(args.ori_info_path)]
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=4, logger=logger, training=False
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, to_cpu=False, logger=logger)
    model.eval().cuda()

    for i, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, _ = model(batch_dict)

        mask = pred_dicts[0]['pred_scores'] > 0.7
        pred_boxes = pred_dicts[0]['pred_boxes'][mask].cpu().numpy()
        pred_labels = pred_dicts[0]['pred_labels'][mask].cpu().numpy() - 1
        self_train_annos = {
            'names': np.array([cfg.CLASS_NAMES[i] for i in pred_labels]),
            'boxes': pred_boxes
        }
        data_info[i]['self_train_annos'] = self_train_annos
        if args.is_unsupervised:
            annos = data_info[i]['annos']
            keep_indices = [i for i, x in enumerate(annos['name']) if x not in ['Car', 'Pedestrian', 'Cyclist']]
            for key in annos.keys():
                if key == 'gt_boxes_lidar':
                    continue
                annos[key] = annos[key][keep_indices]
            data_info[i]['annos'] = annos
        if (i + 1) % 100 == 0:
            logger.info(f'Generate self train info for sample index: {i + 1} / {len(test_set)}')

    with open(args.new_info_path, 'wb') as f:
        pickle.dump(data_info, f)
    logger.info(f'Saved self train info file to {args.new_info_path}')