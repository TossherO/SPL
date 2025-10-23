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


cfg_file = './cfgs/my_models/centerpoint_kitti.yaml'
cfg_from_yaml_file(cfg_file, cfg)
cfg.TAG = Path(cfg_file).stem

logger = common_utils.create_logger('output/log.txt', rank=cfg.LOCAL_RANK)
# pretrained_model = '../output/cfgs/lion_models/lion_mamba_unitr_convnext/default/ckpt/checkpoint_epoch_12.pth'

test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False, workers=4, logger=logger, training=False
)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
# model.load_params_from_file(filename=pretrained_model, to_cpu=False, logger=logger)
model.eval().cuda()

label_colors = [[1, 0, 0], [0, 0.4, 0], [0, 0, 1]]
pcd_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE

if not os.path.exists('output/visualization'):
    os.makedirs('output/visualization')

with torch.no_grad():
    for i, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)   
        preds_dict, _ = model(batch_dict)
        gt_boxes = batch_dict['gt_boxes'][0, :, :7].cpu().numpy()
        gt_labels = batch_dict['gt_boxes'][0, :, -1].cpu().long().numpy()
        pred_boxes = preds_dict[0]['pred_boxes'].cpu().numpy()
        pred_labels = preds_dict[0]['pred_labels'].cpu().numpy()

        # Visualization
        points = batch_dict['points'][0].cpu().numpy()
        plt.figure(figsize=(30, 30))
        plt.xlim(pcd_range[0], pcd_range[3])
        plt.ylim(pcd_range[1], pcd_range[4])
        plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=0.5)

        for j in range(gt_boxes.shape[0]):
            corners2d = bbox3d_to_corners2d(gt_boxes[j])
            plt.plot(corners2d[0], corners2d[1], c=label_colors[gt_labels[j]], linewidth=3)

        for j in range(pred_boxes.shape[0]):
            corners2d = bbox3d_to_corners2d(pred_boxes[j])
            plt.plot(corners2d[0], corners2d[1], c=label_colors[pred_labels[j]], linewidth=3, linestyle='--')

        plt.axis('off')
        plt.savefig(f'output/visualization/sample_{i:03d}.png', bbox_inches='tight', pad_inches=0)
        plt.close()