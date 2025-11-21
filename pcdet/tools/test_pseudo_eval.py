import time
import torch
import torch.nn as nn
from pathlib import Path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
import pickle

cfg_file = './cfgs/my_models/centerpoint_kitti_prototype.yaml'
cfg_from_yaml_file(cfg_file, cfg)
cfg.TAG = Path(cfg_file).stem
cfg.DATA_CONFIG.INFO_PATH['test'] = ['kitti_infos_train_pseudo.pkl']

logger = common_utils.create_logger('output/log.txt', rank=cfg.LOCAL_RANK)

test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=4,
    dist=False, workers=4, logger=logger, training=False
)

dataset = test_loader.dataset
class_names = dataset.class_names
det_annos = []
for i, batch_dict in enumerate(test_loader):
    load_data_to_gpu(batch_dict)
    is_valid = (batch_dict['pseudo_boxes'][:, :, 3] > 0) & (batch_dict['pseudo_boxes'][:, :, 9] > 0)
    pred_scores = batch_dict['pseudo_boxes'][:, :, 8]
    pred_boxes = batch_dict['pseudo_boxes'][:, :, :7]
    # pred_boxes[:, :, 3:6] = batch_dict['pseudo_boxes'][:, :, 10:13]
    pred_labels = batch_dict['pseudo_boxes'][:, :, 7].long()
    pred_dicts = []
    for bs_idx in range(pred_scores.shape[0]):
        pred_dict = {
            'pred_boxes': pred_boxes[bs_idx][is_valid[bs_idx]],
            'pred_scores': pred_scores[bs_idx][is_valid[bs_idx]],
            'pred_labels': pred_labels[bs_idx][is_valid[bs_idx]],
        }
        pred_dicts.append(pred_dict)

    annos = dataset.generate_prediction_dicts(
        batch_dict, pred_dicts, class_names,
        output_path=None
    )
    det_annos += annos    

result_str, result_dict = dataset.evaluation(
    det_annos, class_names,
    eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    output_path=None
)

logger.info(result_str)