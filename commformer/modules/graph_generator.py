
import torch

def build_leader_follower_adj(n_agents, cluster_labels, leader_indices):
    """
    Returns adjacency matrix (n_agents x n_agents) with 1 if i receives from j
    """
    adj = torch.zeros((n_agents, n_agents), dtype=torch.float32)
    for i in range(n_agents):
        leader = leader_indices[cluster_labels[i]]
        adj[i, leader] = 1  # follower receives from leader
        adj[leader, i] = 1  # optional: leader also receives follower info
    return adj  # Used as mask in attention
