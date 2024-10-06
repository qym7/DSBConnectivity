
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.utils import sample

def compute_graph_rate_matrix(
    self,
    pred_X,
    pred_E,
    X_t,
    E_t,
    dt,
    limit_x,
    limit_e,
    node_mask,
    t,
    eta,
    rdb,
    rdb_crit,
    cfg
):
    device = pred_X.device
    # Zero-out non-existing states
    dx = limit_x.shape[-1]
    de = limit_e.shape[-1]

    X_t_label = X_t.argmax(-1, keepdim=True)
    E_t_label = E_t.argmax(-1, keepdim=True)

    if not cfg.sample.x1_parameterization:
        sampled_1 = sample(pred_X, pred_E, onehot=True, node_mask=node_mask)
        X_1_pred = sampled_1.X
        E_1_pred = sampled_1.E

        pc_dt = dt * cfg.sample.guided_weight
        R_t_X, R_t_E, Rstar_t_X, Rstar_t_E, Rdb_t_X, Rdb_t_E, X_mask, E_mask = (
            compute_rate_matrix(
                t,
                eta,
                rdb,
                rdb_crit,
                X_1_pred,
                E_1_pred,
                X_t_label,
                E_t_label,
                pred_X,
                pred_E,
                node_mask,
                return_both=True,
                pc_dt=pc_dt,
            )
        )
    else:
        bs, n, dx = X_t.shape
        # TODO: not efficient at all, I think we can clearly improve the code for rate matrices:
        # 1. we can compute the rate matrices for nodes and edges independently and don't do it jointly at each function
        # 2. I think the x1 parameterization can be made tensorial operations instead of for loops (for Rstar I'm pretty sure, for Rdb I have to think, and for the other operations of zeroing out some entries as well, will do if works)
        dummy_X_1_pred = torch.zeros((bs, n)).long().to(device)
        dummy_E_1_pred = torch.zeros((bs, n, n)).long().to(device)
        # Built R_t_X
        R_t_X_list = []
        dx_no_virtual = False
        # import pdb
        # pdb.set_trace()
        for x_1 in range(dx_no_virtual):
            X_1_pred = x_1 * torch.ones_like(dummy_X_1_pred).long().to(device)
            R_t_X, R_t_E, X_mask, E_mask = compute_rate_matrix(
                t,
                eta,
                rdb,
                rdb_crit,
                X_1_pred,
                dummy_E_1_pred,
                X_t_label,
                E_t_label,
                pred_X,
                pred_E,
                node_mask,
            )
            R_t_X_list.append(R_t_X)
        R_t_X_stacked = torch.stack(R_t_X_list, dim=-1)
        R_t_X = torch.sum(
            R_t_X_stacked * pred_X.unsqueeze(-2), dim=-1
        )  # weight sum

        # Built R_t_E
        de_no_virtual = False
        R_t_E_list = []
        for e_1 in range(de_no_virtual):
            E_1_pred = e_1 * torch.ones_like(dummy_E_1_pred).long().to(device)
            R_t_X, R_t_E, X_mask, E_mask = compute_rate_matrix(
                t,
                eta,
                rdb,
                rdb_crit,
                dummy_X_1_pred,
                E_1_pred,
                X_t_label,
                E_t_label,
                pred_X,
                pred_E,
                node_mask,
            )
            R_t_E_list.append(R_t_E)
        R_t_E_stacked = torch.stack(R_t_E_list, dim=-1)

        R_t_E = torch.sum(
            R_t_E_stacked * pred_E.unsqueeze(-2), dim=-1
        )  # weight sum

    return R_t_X, R_t_E


def compute_rate_matrix(
    t,
    eta,
    rdb,
    rdb_crit,
    X_1_pred,
    E_1_pred,
    X_t_label,
    E_t_label,
    pred_X,
    pred_E,
    node_mask,
    limit_dist,
    return_rdb=False,
    return_rstar=False,
    return_both=False,
    func="relu",
):

    (
        pt_vals_X,
        pt_vals_E,
        pt_vals_at_Xt,
        pt_vals_at_Et,
        dt_p_vals_X,
        dt_p_vals_E,
        dt_p_vals_at_Xt,
        dt_p_vals_at_Et,
    ) = compute_pt_vals(t, X_t_label, E_t_label, X_1_pred, E_1_pred, limit_dist)
    # ) = compute_pt_vals(t, X_column_to_keep, E_column_to_keep, X_1_pred, E_1_pred)

    Rstar_t_X, Rstar_t_E = compute_Rstar(
        X_1_pred,
        E_1_pred,
        X_t_label,
        E_t_label,
        pt_vals_X,
        pt_vals_E,
        pt_vals_at_Xt,
        pt_vals_at_Et,
        dt_p_vals_X,
        dt_p_vals_E,
        dt_p_vals_at_Xt,
        dt_p_vals_at_Et,
        func,
    )

    X_mask, E_mask = compute_RDB(
        pt_vals_X,
        pt_vals_E,
        X_t_label,
        E_t_label,
        pred_X,
        pred_E,
        X_1_pred,
        E_1_pred,
        rdb,
        rdb_crit,
        node_mask,
        t,
    )

    # stochastic rate matrix
    Rdb_t_X = pt_vals_X * X_mask * eta
    Rdb_t_E = pt_vals_E * E_mask * eta

    R_t_X, R_t_E = compute_R(
        Rstar_t_X,
        Rstar_t_E,
        Rdb_t_X,
        Rdb_t_E,
        pt_vals_at_Xt,
        pt_vals_at_Et,
        pt_vals_X,
        pt_vals_E,
        node_mask,
    )

    if return_rstar:
        return R_t_X, R_t_E, Rstar_t_X, Rstar_t_E, X_mask, E_mask

    if return_rdb:
        return R_t_X, R_t_E, Rdb_t_X, Rdb_t_E, X_mask, E_mask

    if return_both:
        return R_t_X, R_t_E, Rstar_t_X, Rstar_t_E, Rdb_t_X, Rdb_t_E, X_mask, E_mask

    return R_t_X, R_t_E, X_mask, E_mask

def compute_Rstar(
    self,
    X_1_pred,
    E_1_pred,
    X_t_label,
    E_t_label,
    pt_vals_X,
    pt_vals_E,
    pt_vals_at_Xt,
    pt_vals_at_Et,
    dt_p_vals_X,
    dt_p_vals_E,
    dt_p_vals_at_Xt,
    dt_p_vals_at_Et,
    func,
):
    # Numerator of R_t^*
    if func == "relu":
        inner_X = dt_p_vals_X - dt_p_vals_at_Xt[:, :, None]
        inner_E = dt_p_vals_E - dt_p_vals_at_Et[:, :, :, None]

        Z_t_X = torch.count_nonzero(pt_vals_X, dim=-1)  # (bs, n)
        Z_t_E = torch.count_nonzero(pt_vals_E, dim=-1)  # (bs, n, n)

        # compensate
        limit_dist = limit_dist.to_device(device)
        X1_onehot = F.one_hot(X_1_pred, num_classes=len(limit_dist.X)).float()
        E1_onehot = F.one_hot(E_1_pred, num_classes=len(limit_dist.E)).float()
        mask_X = X_1_pred.unsqueeze(-1) != X_t_label
        mask_E = E_1_pred.unsqueeze(-1) != E_t_label

        Rstar_t_numer_X = F.relu(inner_X)  # (bs, n, dx)
        Rstar_t_numer_E = F.relu(inner_E)  # (bs, n, n, de)

        # target guidance scheme 2
        Rstar_t_numer_X += X1_onehot * cfg.sample.omega * mask_X
        Rstar_t_numer_E += E1_onehot * cfg.sample.omega * mask_E
    else:
        raise NotImplementedError

    Z_t_X = torch.count_nonzero(pt_vals_X, dim=-1)  # (bs, n)
    Z_t_E = torch.count_nonzero(pt_vals_E, dim=-1)  # (bs, n, n)

    # Denominator of R_t^*
    Rstar_t_denom_X = Z_t_X * pt_vals_at_Xt  # (bs, n)
    Rstar_t_denom_E = Z_t_E * pt_vals_at_Et  # (bs, n, n)
    Rstar_t_X = Rstar_t_numer_X / Rstar_t_denom_X[:, :, None]  # (bs, n, dx)
    Rstar_t_E = Rstar_t_numer_E / Rstar_t_denom_E[:, :, :, None]  # (B, n, n, de)

    Rstar_t_X = torch.nan_to_num(Rstar_t_X, nan=0.0, posinf=0.0, neginf=0.0)
    Rstar_t_E = torch.nan_to_num(Rstar_t_E, nan=0.0, posinf=0.0, neginf=0.0)

    Rstar_t_X[Rstar_t_X > 1e5] = 0.0
    Rstar_t_E[Rstar_t_E > 1e5] = 0.0

    return Rstar_t_X, Rstar_t_E

def compute_RDB(
    self,
    pt_vals_X,
    pt_vals_E,
    X_t_label,
    E_t_label,
    pred_X,
    pred_E,
    X_1_pred,
    E_1_pred,
    rdb,
    rdb_crit,
    node_mask,
    t,
):
    dx = pt_vals_X.shape[-1]
    de = pt_vals_E.shape[-1]
    # Masking Rdb
    if rdb == "general":
        x_mask = torch.ones_like(pt_vals_X)
        e_mask = torch.ones_like(pt_vals_E)
    elif rdb == "marginal":
        x_limit = limit_dist.X
        e_limit = limit_dist.E

        Xt_marginal = x_limit[X_t_label]
        Et_marginal = e_limit[E_t_label]

        x_mask = x_limit.repeat(X_t_label.shape[0], X_t_label.shape[1], 1)
        e_mask = e_limit.repeat(
            E_t_label.shape[0], E_t_label.shape[1], E_t_label.shape[2], 1
        )

        x_mask = x_mask > Xt_marginal
        e_mask = e_mask > Et_marginal

    elif rdb == "column":
        # Get column idx to pick
        if rdb_crit == "max_marginal":
            x_column_idxs = (
                noise_dist.get_limit_dist()
                .X.argmax(keepdim=True)
                .expand(X_t_label.shape)
            )
            e_column_idxs = (
                noise_dist.get_limit_dist()
                .E.argmax(keepdim=True)
                .expand(E_t_label.shape)
            )
        elif rdb_crit == "x_t":
            x_column_idxs = X_t_label
            e_column_idxs = E_t_label
        elif rdb_crit == "abs_state":
            x_column_idxs = torch.ones_like(X_t_label) * (dx - 1)
            e_column_idxs = torch.ones_like(E_t_label) * (de - 1)
        elif rdb_crit == "p_x1_g_xt":
            x_column_idxs = pred_X.argmax(dim=-1, keepdim=True)
            e_column_idxs = pred_E.argmax(dim=-1, keepdim=True)
        elif rdb_crit == "x_1":  # as in paper, uniform
            x_column_idxs = X_1_pred.unsqueeze(-1)
            e_column_idxs = E_1_pred.unsqueeze(-1)
        elif rdb_crit == "p_xt_g_x1":
            x_column_idxs = pt_vals_X.argmax(dim=-1, keepdim=True)
            e_column_idxs = pt_vals_E.argmax(dim=-1, keepdim=True)
        elif rdb_crit == "p_xtdt_g_x0":
            raise ValueError(
                "dt here was not checked after time distorter implementation, please check with YQ before launching"
            )
        elif rdb_crit == "xhat_t":
            sampled_1_hat = sample(
                pt_vals_X,
                pt_vals_E,
                onehot=True
                node_mask=node_mask,
            )
            x_column_idxs = sampled_1_hat.X.unsqueeze(-1)
            e_column_idxs = sampled_1_hat.E.unsqueeze(-1)
        else:
            raise NotImplementedError

        # create mask based on columns picked
        x_mask = F.one_hot(x_column_idxs.squeeze(-1), num_classes=dx)
        x_mask[(x_column_idxs == X_t_label).squeeze(-1)] = 1.0
        e_mask = F.one_hot(e_column_idxs.squeeze(-1), num_classes=de)
        e_mask[(e_column_idxs == E_t_label).squeeze(-1)] = 1.0

    elif rdb == "entry":
        if rdb_crit == "abs_state":
            # select last index
            x_masked_idx = torch.tensor(
                dx
                - 1  # delete -1 for the last index
                # dx - 1
            ).to(
                device
            )  # leaving this for now, can change later if we want to explore it a bit more
            e_masked_idx = torch.tensor(de - 1).to(device)

            x1_idxs = X_1_pred.unsqueeze(-1)  # (bs, n, 1)
            e1_idxs = E_1_pred.unsqueeze(-1)  # (bs, n, n, 1)
        if rdb_crit == "first": # here in all datasets it's the argmax
            # select last index
            x_masked_idx = torch.tensor(0).to(
                device
            )  # leaving this for now, can change later if we want to explore it a bit more
            e_masked_idx = torch.tensor(0).to(device)

            x1_idxs = X_1_pred.unsqueeze(-1)  # (bs, n, 1)
            e1_idxs = E_1_pred.unsqueeze(-1)  # (bs, n, n, 1)
        else:
            raise NotImplementedError

        # create mask based on columns picked
        # bs, n, _ = X_t_label.shape
        # x_mask = torch.zeros((bs, n, dx), device=device)  # (bs, n, dx)
        x_mask = torch.zeros_like(pt_vals_X)  # (bs, n, dx)
        xt_in_x1 = (X_t_label == x1_idxs).squeeze(-1)  # (bs, n, 1)
        x_mask[xt_in_x1] = F.one_hot(x_masked_idx, num_classes=dx).float()
        xt_in_masked = (X_t_label == x_masked_idx).squeeze(-1)
        x_mask[xt_in_masked] = F.one_hot(
            x1_idxs.squeeze(-1), num_classes=dx
        ).float()[xt_in_masked]

        # e_mask = torch.zeros((bs, n, n, de), device=device)  # (bs, n, dx)
        e_mask = torch.zeros_like(pt_vals_E)
        et_in_e1 = (E_t_label == e1_idxs).squeeze(-1)
        e_mask[et_in_e1] = F.one_hot(e_masked_idx, num_classes=de).float()
        et_in_masked = (E_t_label == e_masked_idx).squeeze(-1)
        e_mask[et_in_masked] = F.one_hot(
            e1_idxs.squeeze(-1), num_classes=de
        ).float()[et_in_masked]
    else:
        raise NotImplementedError

    return x_mask, e_mask

def compute_R(
    self,
    Rstar_t_X,
    Rstar_t_E,
    Rdb_t_X,
    Rdb_t_E,
    pt_vals_at_Xt,
    pt_vals_at_Et,
    pt_vals_X,
    pt_vals_E,
    node_mask,
):
    # sum to get the final R_t_X and R_t_E
    R_t_X = Rstar_t_X + Rdb_t_X
    R_t_E = Rstar_t_E + Rdb_t_E

    # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero, which need to be applied to Rdb too
    dx = R_t_X.shape[-1]
    de = R_t_E.shape[-1]
    R_t_X[(pt_vals_at_Xt == 0.0)[:, :, None].repeat(1, 1, dx)] = 0.0
    R_t_E[(pt_vals_at_Et == 0.0)[:, :, :, None].repeat(1, 1, 1, de)] = 0.0
    # zero-out certain columns of R, which is implied in the computation of Rdb
    # if the probability of a place is 0, then we should not consider it in the R computation
    R_t_X[pt_vals_X == 0.0] = 0.0
    R_t_E[pt_vals_E == 0.0] = 0.0

    return R_t_X, R_t_E

def dt_p_xt_g_x1(X1, E1, limit_dist):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    device = X1.device
    
    limit_dist = limit_dist.to_device(device)
    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    dX = X1_onehot - limit_dist.X[None, None, :]
    dE = E1_onehot - limit_dist.E[None, None, None, :]

    assert (dX.sum(-1).abs() < 1e-4).all() and (dE.sum(-1).abs() < 1e-4).all()

    # # very small error will lead to computation issue
    # dX = dX - dX.sum(-1, keepdim=True)
    # dE = dE - dE.sum(-1, keepdim=True)

    return dX, dE

def p_xt_g_x1(X1, E1, t):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    device = X1.device
    t_time = t.squeeze(-1)[:, None, None]
    limit_dist = limit_dist.to_device(device)
    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
    Et = (
        t_time[:, None] * E1_onehot
        + (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
    )

    assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
        (Et.sum(-1) - 1).abs() < 1e-4
    ).all()

    return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)

def compute_pt_vals(t, X_t_label, E_t_label, X_1_pred, E_1_pred, limit_dist):
    dt_p_vals_X, dt_p_vals_E = dt_p_xt_g_x1(
        X_1_pred, E_1_pred, limit_dist
    )  #  (bs, n, dx), (bs, n, n, de)

    dt_p_vals_at_Xt = dt_p_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n, )
    dt_p_vals_at_Et = dt_p_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n, )

    pt_vals_X, pt_vals_E = p_xt_g_x1(
        # X_1_pred, E_1_pred, t + dt
        X_1_pred,
        E_1_pred,
        t,
    )  # (bs, n, dx), (bs, n, n, de)

    pt_vals_at_Xt = pt_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n, )
    pt_vals_at_Et = pt_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n, )

    # print(torch.where(pt_vals_E[..., -1] > 0))
    # import pdb; pdb.set_trace()

    return (
        pt_vals_X,
        pt_vals_E,
        pt_vals_at_Xt,
        pt_vals_at_Et,
        dt_p_vals_X,
        dt_p_vals_E,
        dt_p_vals_at_Xt,
        dt_p_vals_at_Et,
    )
