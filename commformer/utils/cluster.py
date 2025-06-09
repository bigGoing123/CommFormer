# commformer/utils/lf_comm.py

import torch
import numpy as np
from sklearn.cluster import DBSCAN

def cluster_agents(features, eps=5.0, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    n_clusters = max(labels) + 1
    leader_indices = []

    for cid in range(n_clusters):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        center = np.mean(features[members], axis=0)
        leader = members[np.argmin(np.linalg.norm(features[members] - center, axis=1))]
        leader_indices.append(leader)
    return labels, leader_indices

def build_lf_adj(n_agents, labels, leader_indices, device):
    adj = torch.zeros((n_agents, n_agents), device=device)
    for i in range(n_agents):
        if labels[i] == -1 or len(leader_indices) == 0 or labels[i] >= len(leader_indices): # 跳过未分组的智能体，或没有leader的情况
            continue
        leader = leader_indices[labels[i]]
        adj[i, leader] = 1.0
        adj[leader, i] = 1.0  # 双向通信，可修改
    return adj  # shape: n x n
