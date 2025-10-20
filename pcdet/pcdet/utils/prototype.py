import os
import numpy as np
import torch
from pcdet.utils.kmeans import kmeans


def load_prototype(prototype_cfg, load_path=None):
    prototype_path = prototype_cfg.get('PATH', None) if load_path is None else load_path
    if os.path.exists(prototype_path):
        prototype_data = torch.load(prototype_path, map_location='cpu')
        proto_features = prototype_data['proto_features'].cuda()  # (num_class, c, d)
        feature_bank = prototype_data['feature_bank'].cuda()  # (num_class, k, d)
        feature_count = prototype_data['feature_count'].cuda()  # (num_class, )
        print(f'Prototype loaded from {prototype_path}')
        assert proto_features.shape == (prototype_cfg.NUM_CLASS, prototype_cfg.NUM_PROTO, prototype_cfg.FEATURE_DIM), \
            f'Wrong proto_features shape {proto_features.shape}, ' \
            f'expected ({prototype_cfg.NUM_CLASS}, {prototype_cfg.NUM_PROTO}, {prototype_cfg.FEATURE_DIM})'
        assert feature_bank.shape == (prototype_cfg.NUM_CLASS, prototype_cfg.BANK_SIZE, prototype_cfg.FEATURE_DIM), \
            f'Wrong feature_bank shape {feature_bank.shape}, ' \
            f'expected ({prototype_cfg.NUM_CLASS}, {prototype_cfg.BANK_SIZE}, {prototype_cfg.FEATURE_DIM})'
        assert feature_count.shape == (prototype_cfg.NUM_CLASS, ), \
            f'Wrong feature_count shape {feature_count.shape}, expected ({prototype_cfg.NUM_CLASS}, )'
    else:
        proto_features = torch.zeros((prototype_cfg.NUM_CLASS, prototype_cfg.NUM_PROTO, prototype_cfg.FEATURE_DIM)).float().cuda()
        feature_bank = torch.zeros((prototype_cfg.NUM_CLASS, prototype_cfg.BANK_SIZE, prototype_cfg.FEATURE_DIM)).float().cuda()
        feature_count = torch.zeros((prototype_cfg.NUM_CLASS, )).long().cuda()
        print(f'Prototype path {prototype_path} does not exist!')
    return proto_features, feature_bank, feature_count


def save_prototype(proto_features, feature_bank, feature_count, prototype_cfg, save_path=None):
    prototype_path = prototype_cfg.get('PATH', None) if save_path is None else save_path
    prototype_data = {
        'proto_features': proto_features,
        'feature_bank': feature_bank,
        'feature_count': feature_count,
    }
    torch.save(prototype_data, prototype_path)
    print(f'Prototype saved to {prototype_path}')


def prototype_update(proto_features, feature_bank, feature_count, new_features, num_prototype):
    """
    Args:
        proto_features: (num_class, num_prototype, feature_dim)
        feature_bank: (num_class, bank_size, feature_dim)
        feature_count: (num_class, )
        new_features: List of (num_new, feature_dim) tensors, len = num_class
        num_prototypes: int
    """
    num_class, bank_size, feature_dim = feature_bank.shape
    feat2proto_count = torch.zeros((num_class, num_prototype)).long().cuda()

    for class_id in range(num_class):
        if len(new_features[class_id]) == 0:
            continue
        features = torch.cat([feature_bank[class_id, :feature_count[class_id]], new_features[class_id]], dim=0)  # (num_old + num_new, feature_dim)
        
        if len(features) < num_prototype:
            proto_features[class_id, :len(features)] = features
            feat2proto_count[class_id, :len(features)] = 1

        else:
            if len(features) > bank_size:
                # 合并相似度最高的特征，直到数量不超过 bank_size
                num_merge = len(features) - bank_size
                cosine_sim = torch.matmul(features, features.T)  # (N, N)
                cosine_sim = torch.triu(cosine_sim, diagonal=1)  # 只考虑上三角，避免重复计算
                for _ in range(num_merge):
                    max_sim_idx = torch.argmax(cosine_sim).item()
                    i, j = divmod(max_sim_idx, len(features))
                    # 合并 i 和 j
                    merged_feature = features[i] + features[j]
                    merged_feature = merged_feature / torch.norm(merged_feature, p=2)  # 归一化
                    # 创建新的特征矩阵（移除了j，用合并后的向量替换i）
                    indices = torch.arange(len(features), device=features.device)
                    keep_indices = indices[(indices != i) & (indices != j)]
                    features = torch.cat([features[keep_indices], merged_feature.unsqueeze(0)], dim=0)
                    # 创建新的余弦相似度矩阵
                    new_cosine_sim = torch.zeros((len(features), len(features)), device=features.device)
                    new_cosine_sim[:-1, :-1] = cosine_sim[keep_indices][:, keep_indices]
                    new_cosine_sim[:-1, -1] = torch.matmul(features[:-1], merged_feature)
                    cosine_sim = new_cosine_sim

            # kmeans to get new prototypes
            feat2proto, new_proto = kmeans(features, num_prototype, distance='cosine', norm=False)  # (num_prototype, feature_dim)
            proto_features[class_id] = new_proto
            feat2proto_count[class_id] = torch.bincount(feat2proto, minlength=num_prototype)

        feature_bank[class_id, :len(features)] = features
        feature_count[class_id] = len(features)

    return proto_features, feature_bank, feature_count, feat2proto_count