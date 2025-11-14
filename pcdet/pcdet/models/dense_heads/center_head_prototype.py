import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from timm.layers import trunc_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from functools import partial


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, norm_func=None):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead_prototype(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        norm_func = partial(nn.BatchNorm2d, eps=self.model_cfg.get('BN_EPS', 1e-5), momentum=self.model_cfg.get('BN_MOM', 0.1))
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            norm_func(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    norm_func=norm_func
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

        # Prototype related
        self.prototype_cfg = self.model_cfg.get('PROTOTYPE_CONFIG', None)
        self.proto_features = nn.Parameter(torch.zeros(
            (self.prototype_cfg.NUM_CLASS, self.prototype_cfg.NUM_PROTO, self.prototype_cfg.FEATURE_DIM)).float(), requires_grad=False)
        self.bank_labels = nn.Parameter(torch.zeros((
            self.prototype_cfg.NUM_CLASS, self.prototype_cfg.BANK_SIZE)).long(), requires_grad=False)
        self.proto_proj = nn.Sequential(
            nn.Conv2d(
                self.model_cfg.SHARED_CONV_CHANNEL, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            norm_func(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
            nn.Conv2d(
                self.model_cfg.SHARED_CONV_CHANNEL, self.prototype_cfg.FEATURE_DIM, 1, stride=1, padding=0,
                bias=True
            ),
        )
        self.proto_proj_norm = nn.LayerNorm(self.prototype_cfg.FEATURE_DIM)
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.memory_features = [torch.zeros((0, self.prototype_cfg.FEATURE_DIM)).float().cuda() for _ in range(self.prototype_cfg.NUM_CLASS)]
        self.new_features = None  # list of (M_c, D), len = NUM_CLASS
        self.new_features_labels = None  # list of (N_c,), len = NUM_CLASS
        self.proto_feature_thre = self.prototype_cfg.PROTO_FEATURE_THRESHOLD
        self.pseudo_label_thre = self.prototype_cfg.PSEUDO_LABEL_THRESHOLD
        self.max_proto_update_num = self.prototype_cfg.MAX_PROTO_UPDATE_NUM
        self.proto_stage = 0
        # trunc_normal_(self.proto_features, std=0.02)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        self.refresh_new_features()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask, ret_boxes_src

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'target_boxes_src': [],
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, target_boxes_src_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, ret_boxes_src = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                target_boxes_src_list.append(ret_boxes_src.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['target_boxes_src'].append(torch.stack(target_boxes_src_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            # hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx], mask=target_dicts['heatmaps_mask'][idx])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            if 'iou' in pred_dict or self.model_cfg.get('IOU_REG_LOSS', False):

                batch_box_preds = centernet_utils.decode_bbox_from_pred_dicts(
                    pred_dict=pred_dict,
                    point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                    feature_map_stride=self.feature_map_stride
                )  # (B, H, W, 7 or 9)

                if 'iou' in pred_dict:
                    batch_box_preds_for_iou = batch_box_preds.permute(0, 3, 1, 2)  # (B, 7 or 9, H, W)

                    iou_loss = loss_utils.calculate_iou_loss_centerhead(
                        iou_preds=pred_dict['iou'],
                        batch_box_preds=batch_box_preds_for_iou.clone().detach(),
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    loss += iou_loss
                    tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()

                if self.model_cfg.get('IOU_REG_LOSS', False):
                    iou_reg_loss = loss_utils.calculate_iou_reg_loss_centerhead(
                        batch_box_preds=batch_box_preds_for_iou,
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    if target_dicts['masks'][idx].sum().item() != 0:
                        iou_reg_loss = iou_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                        loss += iou_reg_loss
                        tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
                    else:
                        loss += (batch_box_preds_for_iou * 0.).sum()
                        tb_dict['iou_reg_loss_head_%d' % idx] = (batch_box_preds_for_iou * 0.).sum()

        # Constrastive loss for prototype learning
        logit_scale = self.logit_scale.exp()
        for class_id in range(self.prototype_cfg.NUM_CLASS):
            if self.new_features[class_id].shape[0] == 0:
                continue
            feats = self.new_features[class_id]  # (N_c, D)

            if self.proto_stage > 0:
                labels = self.new_features_labels[class_id]  # (N_c,)
                logits_intra = logit_scale * feats @ self.proto_features[class_id].t()  # (N_c, NUM_PROTO)
                contrastive_intra_loss = F.cross_entropy(logits_intra, labels)
                proto_features_inter = torch.cat([
                    self.proto_features[cid] for cid in range(self.prototype_cfg.NUM_CLASS) if cid != class_id
                ], dim=0)  # (sum(NUM_PROTO_not_c), D)
                logits_inter = logit_scale * feats @ proto_features_inter.t()  # (N_c, sum(NUM_PROTO_not_c))
                logits_inter = torch.cat([
                    logits_intra[torch.arange(feats.shape[0]), labels].unsqueeze(1), logits_inter
                ], dim=1)  # (N_c, 1 + sum(NUM_PROTO_not_c))
                contrastive_inter_loss = F.cross_entropy(logits_inter, torch.zeros_like(labels).long())
            else:
                if self.memory_features[class_id].shape[0] == 0:
                    loss += (feats * 0.).sum() * logit_scale
                    continue
                pos_features = self.memory_features[class_id]  # (M_c, D)
                pos_logits = feats @ pos_features.t()  # (N_c, M_c)
                max_pos_logit, _ = pos_logits.max(dim=1)  # (N_c,)
                neg_features = torch.cat([
                    self.memory_features[cid] for cid in range(self.prototype_cfg.NUM_CLASS) if cid != class_id
                ], dim=0)  # (sum(M_not_c), D)
                neg_logits = feats @ neg_features.t()  # (N_c, sum(M_not_c))
                logits = logit_scale * torch.cat([max_pos_logit.unsqueeze(1), neg_logits], dim=1)  # (N_c, 1 + sum(M_not_c))
                labels = torch.zeros((feats.shape[0],), dtype=torch.long).cuda()  # (N_c,)
                contrastive_intra_loss = torch.tensor(0.).cuda()
                contrastive_inter_loss = F.cross_entropy(logits, labels)
                
            loss += contrastive_intra_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['contrastive_intra_weight'] / self.prototype_cfg.NUM_CLASS
            loss += contrastive_inter_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['contrastive_inter_weight'] / self.prototype_cfg.NUM_CLASS
            tb_dict['contrastive_intra_loss_class_%d' % class_id] = contrastive_intra_loss.item()
            tb_dict['contrastive_inter_loss_class_%d' % class_id] = contrastive_inter_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_iou' in final_dict:
                    pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
                    IOU_RECTIFIER = final_dict['pred_scores'].new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'], 1 - IOU_RECTIFIER[final_dict['pred_labels']]) * torch.pow(pred_iou, IOU_RECTIFIER[final_dict['pred_labels']])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE not in  ['circle_nms', 'class_specific_nms']:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            # if self.proto_stage > 1:
            #     pseudo_heatmap_limit = self.get_preprocess_pseudo_targets(
            #         data_dict['pseudo_boxes'], feature_map_size=spatial_features_2d.size()[2:]
            #     )  # (B, NUM_CLASS, H, W)
            # else:
            pseudo_heatmap_limit = None
            
            # for test
            # target_dict['ori_heatmaps'] = [h.clone() for h in target_dict['heatmaps']]
            # target_dict['pseudo_heatmap_limit'] = pseudo_heatmap_limit

            target_dict = self.prototype_learning(x, pred_dicts, target_dict, pseudo_heatmap_limit)
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict

    def refresh_new_features(self):
        self.new_features = [torch.zeros((0, self.prototype_cfg.FEATURE_DIM)).float().cuda() for _ in range(self.prototype_cfg.NUM_CLASS)]
        self.new_features_labels = [torch.zeros((0,)).long().cuda() for _ in range(self.prototype_cfg.NUM_CLASS)]

    def get_preprocess_pseudo_targets(self, pseudo_boxes, feature_map_size,
                                       gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            pseudo_boxes: list of (B, M, 13)
            feature_map_size: (2) [H, W]
        Returns:
            pre_pseudo_masks: (B, NUM_CLASS, H, W)
        """
        batch_size = pseudo_boxes.shape[0]
        pseudo_heatmap_limit = torch.zeros(
            (batch_size, self.prototype_cfg.NUM_CLASS, feature_map_size[0], feature_map_size[1])
        ).float().cuda()
        cls_ids = (pseudo_boxes[:, :, 7] - 1).long()

        x, y, z = pseudo_boxes[:, :, 0], pseudo_boxes[:, :, 1], pseudo_boxes[:, :, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[1] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[0] - 0.5)
        center = torch.cat((coord_x[:, :, None], coord_y[:, :, None]), dim=-1)
        # center_int = center.int()

        dx, dy, dz = pseudo_boxes[:, :, 3], pseudo_boxes[:, :, 4], pseudo_boxes[:, :, 5]
        is_bbox3d = dx > 0
        invalid_car = ~is_bbox3d & (cls_ids == self.class_names.index('Car'))
        dx[invalid_car], dy[invalid_car] = 2.5, 1.2
        invalid_ped = ~is_bbox3d & (cls_ids == self.class_names.index('Pedestrian'))
        dx[invalid_ped], dy[invalid_ped] = 0.6, 0.5
        invalid_cyc = ~is_bbox3d & (cls_ids == self.class_names.index('Cyclist'))
        dx[invalid_cyc], dy[invalid_cyc] = 1.2, 0.5
        dx = dx / self.voxel_size[0] / self.feature_map_stride
        dy = dy / self.voxel_size[1] / self.feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for bs_idx in range(batch_size):
            for obj_idx in range(pseudo_boxes.shape[1]):
                cur_class_id = cls_ids[bs_idx, obj_idx]
                if cur_class_id < 0 or cur_class_id >= self.prototype_cfg.NUM_CLASS:
                    continue
                centernet_utils.draw_gaussian_to_heatmap(
                    pseudo_heatmap_limit[bs_idx, cur_class_id],
                    center[bs_idx, obj_idx], radius[bs_idx, obj_idx].item()
                )
            
        return pseudo_heatmap_limit


    def prototype_learning(self, x, pred_dicts, target_dict, pseudo_heatmap_limit):
        """
        Args:
            x: (B, D, H, W)
            pred_dicts: list of dict
                each dict contains:
                    'hm': (B, num_classes, H, W)
                    'center': (B, 2, H, W)
                    'center_z': (B, 1, H, W)
                    'dim': (B, 3, H, W)
                    'rot': (B, 2, H, W)
            target_dict:
                'heatmaps': list of (B, num_classes, H, W)
                'target_boxes': list of (B, M, 8)
                'inds': list of (B, M)
                'masks': list of (B, M)
                'target_boxes_src': list of (B, M, 9)
            pseudo_heatmap_limit: (B, NUM_CLASS, H, W)
        Returns:

        """
        # feature extraction
        feats = self.proto_proj(x).permute(0, 2, 3, 1)  # (B, H, W, D)
        feats = F.normalize(self.proto_proj_norm(feats), p=2, dim=-1)

        # compute cosine similarity and assign prototypes without tracking gradients
        if self.proto_stage > 0:
            with torch.no_grad():
                feats_detached = feats.detach()
                cos_sim = torch.einsum('bhwd,cpd->bhwcp', feats_detached, self.proto_features)  # (B, H, W, NUM_CLASS, NUM_PROTO)
                max_sim, max_proto_ids = cos_sim.max(dim=-1)  # (B, H, W, NUM_CLASS)
                feat2proto_count = []
                for class_id in range(self.prototype_cfg.NUM_CLASS):
                    feat2proto_count.append(torch.bincount(
                        self.bank_labels[class_id], minlength=self.prototype_cfg.NUM_PROTO + 1
                    )[1:])  # (NUM_PROTO,)
                feat2proto_count = torch.stack(feat2proto_count, dim=0) # (NUM_CLASS, NUM_PROTO)
                proto_scale = (feat2proto_count.float() + 1).log().clamp(min=1.0) # (NUM_CLASS, NUM_PROTO)
                cos_sim /= proto_scale[None, None, None, :, :]
                max_sim_rect, max_proto_ids_rect = cos_sim.max(dim=-1)  # (B, H, W, NUM_CLASS)
        
        for idx, cur_class_ids in enumerate(self.class_id_mapping_each_head):
            heatmaps = target_dict['heatmaps'][idx]  # (B, num_classes, H, W)

            # collect gt features for prototype updating
            gt_inds = torch.where(heatmaps == 1)  # (4, M) (B, class_id, y, x)
            gt_feats = feats[gt_inds[0], gt_inds[2], gt_inds[3], :]  # (M, D)
            gt_class_ids = cur_class_ids[gt_inds[1]]  # (M,)
            if self.proto_stage > 0:
                gt_proto_ids = max_proto_ids_rect[gt_inds[0], gt_inds[2], gt_inds[3], gt_class_ids]  # (M,)
            for class_id in cur_class_ids:
                class_mask = (gt_class_ids == class_id)
                if class_mask.sum() == 0:
                    continue
                class_feats = gt_feats[class_mask]  # (N_c, D)
                self.new_features[class_id] = torch.cat([self.new_features[class_id], class_feats], dim=0)
                if self.proto_stage > 0:
                    self.new_features_labels[class_id] = torch.cat([self.new_features_labels[class_id], gt_proto_ids[class_mask]], dim=0)
                else:
                    self.new_features_labels[class_id] = torch.cat([self.new_features_labels[class_id], torch.zeros_like(class_feats[:, 0]).long()], dim=0)

            # remove gt positions from max_sim to avoid duplicate counting
            if self.proto_stage > 0:
                heatmap_inds = torch.where(heatmaps > 0)  # (4, N) (B, class_id, y, x)
                max_sim[heatmap_inds[0], heatmap_inds[2], heatmap_inds[3], cur_class_ids[heatmap_inds[1]]] = 0

        if self.proto_stage > 1:
            max_sim, max_class_ids = max_sim.max(dim=-1)  # (B, H, W)

            # sim_mask = (max_sim > self.proto_feature_thre)
            # for class_id in range(self.prototype_cfg.NUM_CLASS):
            #     class_mask = (max_class_ids == class_id) & sim_mask  # (B, H, W)
            #     # class_mask = (max_class_ids == class_id) & sim_mask & (pseudo_heatmap_limit[:, class_id, :, :] > 0)  # (B, H, W)
            #     if class_mask.sum() == 0:
            #         continue
            #     class_feats = feats[class_mask]  # (N_c, D)
            #     remain_num = self.max_proto_update_num - self.new_features[class_id].shape[0]
            #     if remain_num > 0:
            #         sim_scores = max_sim[class_mask]  # (N_c,)
            #         # sample_ids = torch.randperm(class_feats.shape[0])[:min(remain_num, class_feats.shape[0])]
            #         sample_ids = torch.argsort(sim_scores, descending=True)[:min(remain_num, class_feats.shape[0])]
            #         self.new_features[class_id] = torch.cat([self.new_features[class_id], class_feats[sample_ids]], dim=0)
            #         self.new_features_labels[class_id] = torch.cat([self.new_features_labels[class_id], max_proto_ids_rect[class_mask][:, class_id][sample_ids]], dim=0)

            sim_mask = (max_sim > self.proto_feature_thre)
            pseudo_mask = (max_sim > self.pseudo_label_thre)
            pred_hm_mask = [pred_dict['hm'].sigmoid() > 0.2 for pred_dict in pred_dicts]
            target_dict['heatmaps_mask'] = [torch.ones_like(h[:, 0, :, :]) for h in target_dict['heatmaps']]  # list of (B, H, W)
            
            for idx, cur_class_ids in enumerate(self.class_id_mapping_each_head):
                pseudo_heatmap = torch.zeros_like(target_dict['heatmaps'][idx])  # (B, num_classes, H, W)
                for i, class_id in enumerate(cur_class_ids):

                    # collect similar features for prototype updating
                    class_mask = (max_class_ids == class_id) & sim_mask & pred_hm_mask[idx][:, i, :, :]  # (B, H, W)
                    if class_mask.sum() > 0:
                        class_feats = feats[class_mask]  # (N_c, D)
                        remain_num = self.max_proto_update_num - self.new_features[class_id].shape[0]
                        if remain_num > 0:
                            sim_scores = max_sim[class_mask]  # (N_c,)
                            # sample_ids = torch.randperm(class_feats.shape[0])[:min(remain_num, class_feats.shape[0])]
                            sample_ids = torch.argsort(sim_scores, descending=True)[:min(remain_num, class_feats.shape[0])]
                            self.new_features[class_id] = torch.cat([self.new_features[class_id], class_feats[sample_ids]], dim=0)
                            self.new_features_labels[class_id] = torch.cat([self.new_features_labels[class_id], max_proto_ids_rect[class_mask][:, class_id][sample_ids]], dim=0)
                    
                    # collect pseudo-labeled features and refine heatmaps
                    class_mask = (max_class_ids == class_id) & pseudo_mask & pred_hm_mask[idx][:, i, :, :]  # (B, H, W)
                    # class_mask = (max_class_ids == class_id) & pseudo_mask & (pseudo_heatmap_limit[:, class_id, :, :] > 0)  # (B, H, W)
                    pseudo_heatmap[:, i, :, :][class_mask] = max_sim[class_mask]
                    # target_dict['heatmaps_mask'][idx][pseudo_mask | (pseudo_heatmap_limit[:, class_id, :, :] > 0)] = 0
                
                target_dict['heatmaps'][idx] = torch.clamp_max(target_dict['heatmaps'][idx] + pseudo_heatmap, max=1.0)
                # target_dict['heatmaps_mask'][idx][torch.sum(target_dict['heatmaps'][idx], dim=1) > 0] = 1
        else:
            target_dict['heatmaps_mask'] = [None for _ in target_dict['heatmaps']]

        return target_dict