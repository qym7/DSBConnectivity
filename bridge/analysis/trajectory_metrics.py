import torch

from .. import utils

def ce_trajectory(source, target, n_nodes):
    
    import pdb; pdb.set_trace()

def abs_trajectory(source, target, n_nodes, virtual=False):
    bs, max_n_nodes = source.X.shape
    device = source.X.device
    
    node_mask, edge_mask = utils.get_masks(n_nodes, max_n_nodes, bs, device)
    # virtual nodes typically introduce more entropy for edges since virtual nodes only have non-existing edges
    # what is the good solution here?
    if virtual:
        pass
        # virtual_n_nodes = (source.X > 0).sum(-1)
        # _, edge_mask = utils.get_masks(virtual_n_nodes, max_n_nodes, bs, device)
    
    X_abs = ((source.X != target.X) * node_mask).sum() / 2
    E_abs = ((source.E != target.E) * edge_mask).sum() / 2

    return X_abs.sum(), E_abs.sum(), node_mask, edge_mask


def accumulated_abs_trajectory(trajectory, n_nodes, virtual=False):
    _, len_traj, _, _ = trajectory.X.shape
    
    X_nbr_changes = 0
    E_nbr_changes = 0
    X_ratio_changes = 0
    E_ratio_changes = 0

    for i in range(len_traj-1):
        tource = trajectory.get_data(i, dim=1).collapse()
        target = trajectory.get_data(i+1, dim=1).collapse()
        X_abs, E_abs, node_mask, edge_mask = abs_trajectory(tource, target, n_nodes, virtual=virtual)

        X_nbr_changes += X_abs
        E_nbr_changes += E_abs
        X_ratio_changes += X_abs / node_mask.sum()
        E_ratio_changes += E_abs / edge_mask.sum()

    return X_nbr_changes, E_nbr_changes, X_ratio_changes/(len_traj-1), E_ratio_changes/(len_traj-1)

