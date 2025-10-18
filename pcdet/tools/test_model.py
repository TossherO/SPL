import time
import torch
import torch.nn as nn
from pathlib import Path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
from mmdet3d.utils import register_all_modules
import pickle

# info_path = '../data/nuscenes/v1.0-mini/nuscenes_dbinfos_10sweeps_withvelo.pkl'
# with open(info_path, 'rb') as f:
#     infos = pickle.load(f)

# print(infos.keys())
# print(sum([len(infos[k]) for k in infos.keys()]))

# cfg_file = './cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim_ep48.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba2_nus.yaml'
# cfg_file = './cfgs/lion_models/lion_bimamba_nus.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba_stream.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba_stream2.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba_stream3.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba_stream3.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba2_stream.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba_unitr_nus3.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba_unitr3_stream.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba2_unitr.yaml'
cfg_file = './cfgs/lion_models/lion_mamba_unitr_convnext.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba_unitr_swint.yaml'
# cfg_file = './cfgs/lion_models/lion_mamba_nus_tta.yaml'
cfg_from_yaml_file(cfg_file, cfg)
cfg.TAG = Path(cfg_file).stem
# pretrained_model = '../output/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim_ep48/default/ckpt/checkpoint_epoch_48.pth'
# pretrained_model = '../output/cfgs/lion_models/lion_mamba2_nus/default/ckpt/checkpoint_epoch_48.pth'
# pretrained_model = '../output/cfgs/lion_models/lion_mamba_stream2/default/checkpoint_iter_42174.pth'
# pretrained_model = '../output/cfgs/lion_models/lion_mamba_unitr_nus3/default/ckpt/checkpoint_epoch_44.pth'
pretrained_model = '../output/cfgs/lion_models/lion_mamba_unitr_convnext/default/ckpt/checkpoint_epoch_12.pth'
# pretrained_model = '../output/cfgs/lion_models/lion_mamba_unitr_swint/default/ckpt/checkpoint_epoch_12.pth'

logger = common_utils.create_logger('output/log.txt', rank=cfg.LOCAL_RANK)

cfg.DATA_CONFIG.VERSION = 'v1.0-mini'
cfg.DATA_CONFIG.DATA_PROCESSOR[1]['SHUFFLE_ENABLED']['train'] = False
cfg.DATA_CONFIG.DATA_PROCESSOR[1]['SHUFFLE_ENABLED']['test'] = False

train = False
test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=2,
    dist=False, workers=4, logger=logger, training=train
)
register_all_modules()
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
model.load_params_from_file(filename=pretrained_model, to_cpu=False, logger=logger)
# it, epoch = model.load_params_with_optimizer(filename=pretrained_model, to_cpu=False, optimizer=None, logger=logger)
# print(it, epoch)

if train:
    model.train().cuda()
else:
    model.eval().cuda()

cost_time = 0
for i, batch_dict in enumerate(test_loader):

    load_data_to_gpu(batch_dict)
    # print('gt_boxes:', batch_dict['gt_boxes'][0][:, :6])
    start = time.time()
    if train:
        loss, tb_dict, disp_dict = model(batch_dict)
        # for k in tb_dict:
        #     print(k, tb_dict[k].data)
    else:
        with torch.no_grad():    
            preds_dict, _ = model(batch_dict)
            # print('pred_boxes:', preds_dict[0]['pred_boxes'][:, :6])
            # print('pred_scores:', preds_dict[0]['pred_scores'])
    if i > 0:
        cost_time += time.time() - start

    print(i)
    if i > 29:
        break

print(cost_time)
print('finish!')