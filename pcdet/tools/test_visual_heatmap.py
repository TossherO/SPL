import os
import numpy as np
import torch
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
import matplotlib.pyplot as plt


def bbox3d_to_corners2d(bbox3d):
    x, y, z, l, w, h, yaw = bbox3d
    x1 = x - l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
    y1 = y - l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)
    x2 = x + l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
    y2 = y + l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)
    x3 = x + l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
    y3 = y + l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)
    x4 = x - l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
    y4 = y - l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)
    return np.array([[x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1]])


cfg_file = './cfgs/my_models/centerpoint_kitti_prototype.yaml'
cfg_from_yaml_file(cfg_file, cfg)
cfg.TAG = Path(cfg_file).stem

logger = common_utils.create_logger('output/log.txt', rank=cfg.LOCAL_RANK)
cfg.DATA_CONFIG.INFO_PATH['train'] = ['kitti_infos_train_pseudo.pkl']
cfg.DATA_CONFIG.DATA_AUGMENTOR['AUG_CONFIG_LIST'] = cfg.DATA_CONFIG.DATA_AUGMENTOR['AUG_CONFIG_LIST'][1:]
pretrained_model = '../output/my_models/centerpoint_kitti_prototype/new2_12/ckpt/checkpoint_epoch_40.pth'

test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False, workers=4, logger=logger, training=True
)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
model.load_params_from_file(filename=pretrained_model, to_cpu=False, logger=logger)
model.train().cuda()
model.dense_head.proto_stage = 2

label_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
pcd_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE

if not os.path.exists('output/visualization'):
    os.makedirs('output/visualization')

with torch.no_grad():
    for i, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)   
        loss, tb_dict, disp_dict= model(batch_dict)
        model.dense_head.refresh_new_features()
        gt_boxes = batch_dict['gt_boxes'][0, :, :7].cpu().numpy()
        gt_labels = batch_dict['gt_boxes'][0, :, -1].cpu().long().numpy() - 1
        gt_heatmaps = model.dense_head.forward_ret_dict['target_dicts']['ori_heatmaps'][0][0].cpu().numpy()  # (num_classes, H, W)
        pseudo_heatmap_limit = model.dense_head.forward_ret_dict['target_dicts']['pseudo_heatmap_limit'][0].cpu().numpy()  # (num_classes, H, W)
        pseudo_heatmaps = model.dense_head.forward_ret_dict['target_dicts']['heatmaps'][0][0].cpu().numpy()  # (num_classes, H, W)
        heatmaps_mask = model.dense_head.forward_ret_dict['target_dicts']['heatmaps_mask'][0][0].cpu().numpy()  # (H, W)
        pred_heatmaps = model.dense_head.forward_ret_dict['pred_dicts'][0]['hm'][0].cpu().numpy()  # (num_classes, H, W)
        
        # Visualization
        points = batch_dict['points'].cpu().numpy()
        plt.figure(figsize=(30, 30))
        plt.scatter(points[:, 1], points[:, 2], c=points[:, 3], cmap='viridis', s=0.5)
        for j in range(gt_boxes.shape[0]):
            corners2d = bbox3d_to_corners2d(gt_boxes[j])
            plt.plot(corners2d[0], corners2d[1], c=label_colors[gt_labels[j]], linewidth=2)
        plt.xlim(pcd_range[0], pcd_range[3])
        plt.ylim(pcd_range[1], pcd_range[4])
        plt.axis('off')
        plt.savefig(f'output/visualization/sample_{i:03d}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        H, W = gt_heatmaps.shape[1], gt_heatmaps.shape[2]
        for cls_id in range(len(cfg.CLASS_NAMES)):
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.title(f'GT Heatmap - Class {cls_id}')
            plt.imshow(gt_heatmaps[cls_id], cmap='hot', vmin=0, vmax=1)
            plt.colorbar()
            plt.subplot(1, 4, 2)
            plt.title(f'Pseudo Heatmap Limit - Class {cls_id}')
            plt.imshow(pseudo_heatmap_limit[cls_id], cmap='hot', vmin=0, vmax=1)
            plt.colorbar()
            plt.subplot(1, 4, 3)
            plt.title(f'Pseudo Heatmap - Class {cls_id}')
            plt.imshow(pseudo_heatmaps[cls_id], cmap='hot', vmin=0, vmax=1)
            plt.colorbar()
            plt.subplot(1, 4, 4)
            # plt.title(f'Heatmaps Mask - Class {cls_id}')
            # plt.imshow(heatmaps_mask, cmap='hot', vmin=0, vmax=1)
            plt.title(f'Predicted Heatmap - Class {cls_id}')
            plt.imshow(pred_heatmaps[cls_id], cmap='hot', vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(f'output/visualization/sample_{i:03d}_class_{cls_id}.png', bbox_inches='tight')
            plt.close()
            
        if i >= 9:
            break