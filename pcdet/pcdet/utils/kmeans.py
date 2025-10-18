import numpy as np
import torch
from tqdm import tqdm


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        norm=True,
        display=False,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param display: (bool) display progress [default: True]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # initialize
    initial_state = initialize(X, num_clusters)

    if display:
        iteration = 0
        tqdm_meter = tqdm(desc='[running kmeans]')

    while True:
        if distance == 'cosine':
            dis = pairwise_distance_function(X, initial_state, norm=norm)
        else:
            dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)
        
        # # Use one-hot encoding to create a mask for each cluster
        # mask = torch.nn.functional.one_hot(choice_cluster, num_clusters).float()  # [N, K]
        # # Sum points in each cluster
        # cluster_sums = mask.T @ X  # [K, D]
        # # Count points in each cluster
        # cluster_counts = mask.sum(dim=0).unsqueeze(1)  # [K, 1]
        # # Avoid division by zero
        # cluster_counts = cluster_counts.clamp(min=1)
        # # Update cluster centers
        # initial_state = cluster_sums / cluster_counts
                        
        if distance == 'cosine' and not norm:
            initial_state = initial_state / initial_state.norm(dim=-1, keepdim=True)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

        if display:
            # increment iteration
            iteration = iteration + 1

            # update tqdm meter
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()

        if center_shift ** 2 < tol:
            break

    return choice_cluster, initial_state


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster


def pairwise_distance(data1, data2):
    A = data1.unsqueeze(dim=1) # N*1*M
    B = data2.unsqueeze(dim=0) # 1*N*M
    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, norm=True):
    A = data1.unsqueeze(dim=1) # N*1*M
    B = data2.unsqueeze(dim=0) # 1*N*M

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    if norm:
        A = A / A.norm(dim=-1, keepdim=True)
        B = B / B.norm(dim=-1, keepdim=True)

    cosine = A * B
    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

