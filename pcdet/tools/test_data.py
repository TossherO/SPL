import torch
import torch.nn.functional as F
import time

def merge_vectors_parallel(features: torch.Tensor, M: int) -> torch.Tensor:
    """
    通过计算余弦相似度并行合并最相似的向量对，将特征向量从N减少到M
    
    参数:
        features: 归一化后的特征向量矩阵，形状为[N, C]
        M: 目标特征向量数量
    
    返回:
        合并后的特征向量矩阵，形状为[M, C]
    """
    assert features.dim() == 2, "特征向量应该是2维矩阵"
    N, C = features.shape
    assert M < N, "M必须小于N"
    
    # 如果需要合并的数量很大，采用迭代合并策略
    num_merges_needed = N - M
    current_features = features.clone()
    current_count = N
    
    while current_count > M:
        # 计算当前特征向量之间的余弦相似度矩阵
        similarity_matrix = torch.mm(current_features, current_features.t())
        
        # 排除自相似性（将对角线设置为非常小的值）
        similarity_matrix = similarity_matrix - torch.eye(current_count, device=features.device) * 2
        
        # 找到最相似的对（上三角部分）
        triu_mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
        similarities = similarity_matrix[triu_mask]
        max_sim, max_idx = torch.max(similarities, dim=0)
        
        # 将扁平索引转换为矩阵坐标
        rows = torch.arange(current_count, device=features.device)
        cols = torch.arange(current_count, device=features.device)
        row_col_pairs = torch.cartesian_prod(rows, cols)
        upper_triangular_indices = row_col_pairs[triu_mask.flatten()]
        i, j = upper_triangular_indices[max_idx]
        
        # 合并最相似的一对向量（取平均值）
        merged_vector = (current_features[i] + current_features[j])
        merged_vector = merged_vector / torch.norm(merged_vector, p=2)  # 归一化

        # 创建新的特征矩阵（移除了j，用合并后的向量替换i）
        indices = torch.arange(current_count, device=features.device)
        keep_indices = indices[(indices != i) & (indices != j)]
        new_features = torch.cat([
            current_features[keep_indices],
            merged_vector.unsqueeze(0)
        ], dim=0)
        
        current_features = new_features
        current_count -= 1
    
    return current_features

def efficient_merge_vectors(features: torch.Tensor, M: int, batch_size: int = None) -> torch.Tensor:
    """
    更高效的实现，批量处理相似度计算和合并操作
    
    参数:
        features: 归一化后的特征向量矩阵，形状为[N, C]
        M: 目标特征向量数量
        batch_size: 批处理大小（可选）
    
    返回:
        合并后的特征向量矩阵，形状为[M, C]
    """
    N, C = features.shape
    if batch_size is None:
        batch_size = min(1000, N)  # 默认批处理大小
    
    current_features = features.clone()
    current_count = N
    
    while current_count > M:
        # 计算所有向量对之间的余弦相似度
        sim_matrix = torch.mm(current_features, current_features.t())
        
        # 将对角线设置为非常小的值（排除自相似性）
        sim_matrix = sim_matrix - torch.eye(current_count, device=features.device) * 2
        
        # 找到最相似的一对
        max_sim, max_idx = torch.max(sim_matrix.view(-1), dim=0)
        i = max_idx // current_count
        j = max_idx % current_count
        
        # 确保i < j
        if i > j:
            i, j = j, i
        
        # 合并最相似的一对向量
        merged_vector = (current_features[i] + current_features[j])
        merged_vector = merged_vector / torch.norm(merged_vector, p=2)  # 归一化

        # 创建新的特征矩阵（移除了j，用合并后的向量替换i）
        indices = torch.arange(current_count, device=features.device)
        keep_indices = indices[(indices != i) & (indices != j)]
        new_features = torch.cat([
            current_features[keep_indices],
            merged_vector.unsqueeze(0)
        ], dim=0)
        
        current_features = new_features
        current_count -= 1
    
    return current_features


def efficient_merge_vectors2(features: torch.Tensor, M: int, batch_size: int = None) -> torch.Tensor:
    """
    更高效的实现，批量处理相似度计算和合并操作
    
    参数:
        features: 归一化后的特征向量矩阵，形状为[N, C]
        M: 目标特征向量数量
        batch_size: 批处理大小（可选）
    
    返回:
        合并后的特征向量矩阵，形状为[M, C]
    """
    N, C = features.shape
    if batch_size is None:
        batch_size = min(1000, N)  # 默认批处理大小
    
    current_features = features.clone()
    current_count = N
    
    sim_matrix = torch.mm(current_features, current_features.t())
    sim_matrix = torch.triu(sim_matrix, diagonal=1)  # 只考虑上三角部分
    while current_count > M:
        # 计算所有向量对之间的余弦相似度
        # 找到最相似的一对
        max_sim, max_idx = torch.max(sim_matrix.view(-1), dim=0)
        i = max_idx // current_count
        j = max_idx % current_count
        
        # 合并最相似的一对向量
        merged_vector = (current_features[i] + current_features[j])
        merged_vector = merged_vector / torch.norm(merged_vector, p=2)  # 归一化

        # 创建新的特征矩阵（移除了j，用合并后的向量替换i）
        indices = torch.arange(current_count, device=features.device)
        keep_indices = indices[(indices != i) & (indices != j)]
        current_features = torch.cat([current_features[keep_indices], merged_vector.unsqueeze(0)], dim=0)
        # 创建新的余弦相似度矩阵
        new_sim_matrix = torch.zeros((len(current_features), len(current_features)), device=features.device)
        new_sim_matrix[:-1, :-1] = sim_matrix[keep_indices][:, keep_indices]
        new_sim_matrix[:-1, -1] = torch.matmul(current_features[:-1], merged_vector)
        sim_matrix = new_sim_matrix
        current_count -= 1
    
    return current_features


# 示例使用
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    with torch.no_grad():
        torch.manual_seed(42)

        # 创建示例数据：150个归一化后的特征向量，每个维度为256
        N, C, M = 550, 256, 500
        features = F.normalize(torch.randn(N, C), dim=1)
        
        print(f"原始特征向量形状: {features.shape}")
        
        # 方法1：基本并行实现
        merged_features = merge_vectors_parallel(features, M)
        start_time = time.time()
        merged_features = merge_vectors_parallel(features, M)
        time1 = time.time() - start_time
        print(f"方法1合并后特征向量形状: {merged_features.shape}")
        print(f"方法1耗时: {time1:.4f}秒")
        
        # 方法2：高效实现
        merged_features_efficient = efficient_merge_vectors(features, M)
        start_time = time.time()
        merged_features_efficient = efficient_merge_vectors(features, M)
        time2 = time.time() - start_time
        print(f"方法2合并后特征向量形状: {merged_features_efficient.shape}")
        print(f"方法2耗时: {time2:.4f}秒")
        
        # 验证两种方法结果是否一致
        difference = torch.norm(merged_features - merged_features_efficient)
        print(f"两种方法结果差异: {difference:.6f}")

        # 方法3：更高效实现
        merged_features_efficient2 = efficient_merge_vectors2(features, M)
        start_time = time.time()
        merged_features_efficient2 = efficient_merge_vectors2(features, M)
        time3 = time.time() - start_time
        print(f"方法3合并后特征向量形状: {merged_features_efficient2.shape}")
        print(f"方法3耗时: {time3:.4f}秒")

        # 验证两种方法结果是否一致
        difference2 = torch.norm(merged_features - merged_features_efficient2)
        print(f"方法1和方法3结果差异: {difference2:.6f}")