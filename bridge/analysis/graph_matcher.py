import warnings

import pygmtools as pygm
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

pygm.set_backend("pytorch")  # set default backend for pygmtools

def preproc_graph_match(
    X_0_init,
    E_0_init,
    X_T_init,
    E_T_init,
    mask_0,
    mask_T,
    full_edge_0=True,
    full_edge_T=True,
):
    r"""
    dense : PlaceHolder(X, E, y)
    mask : (bsz, n_node)
    X : (bsz, n_node, n_node_feat)
    E : (bsz, n_node, n_node, n_edge_feat)
    y : (bsz, n_node)
    """
    dev = X_0_init.device

    NN = X_0_init.size(1)
    bsz = X_0_init.size(0)
    nef = E_0_init.size(-1)

    if full_edge_0:
        E_0 = E_0_init.argmax(dim=-1) + 1 - torch.eye(E_0_init.shape[1], device=dev)
        E_0 = E_0 * (mask_0.unsqueeze(2) & mask_0.unsqueeze(1)).float()
    else:
        E_0 = E_0_init.argmax(dim=-1)

    if full_edge_T:
        E_T = E_T_init.argmax(dim=-1) + 1 - torch.eye(E_T_init.shape[1], device=dev)
        E_T = E_T * (mask_T.unsqueeze(2) & mask_T.unsqueeze(1)).float()
    else:
        E_T = E_T_init.argmax(dim=-1)

    conn_0, attr_0, ne0 = pygm.utils.dense_to_sparse(E_0)
    conn_T, attr_T, neT = pygm.utils.dense_to_sparse(E_T)

    # make it sparse (remove batch dimension)
    _ = torch.arange(bsz, device=dev).unsqueeze(-1) * NN
    indices = torch.arange(conn_0.size(1), device=dev).unsqueeze(0).repeat(bsz, 1)
    mask = indices < ne0.unsqueeze(1)
    conn_0 = torch.masked_select(conn_0, mask.unsqueeze(2)).view(-1, 2)
    conn_0 += (_ * mask)[mask].unsqueeze(
        -1
    )  # make edge index increasing with batch index
    conn_0 = conn_0.t()
    attr_0 = torch.masked_select(attr_0, mask.unsqueeze(2)).view(-1, 1)
    if full_edge_0:
        attr_0 = attr_0 - 1

    indices = torch.arange(conn_T.size(1), device=dev).unsqueeze(0).repeat(bsz, 1)
    mask = indices < neT.unsqueeze(1)
    conn_T = torch.masked_select(conn_T, mask.unsqueeze(2)).view(-1, 2)
    conn_T += (_ * mask)[mask].unsqueeze(
        -1
    )  # make edge index increasing with batch index
    conn_T = conn_T.t()
    attr_T = torch.masked_select(attr_T, mask.unsqueeze(2)).view(-1, 1)
    if full_edge_T:
        attr_T = attr_T - 1

    attr_0 = F.one_hot(attr_0.squeeze(-1).long(), num_classes=nef).float()
    attr_T = F.one_hot(attr_T.squeeze(-1).long(), num_classes=nef).float()

    E_0 = E_0_init
    E_T = E_T_init

    nn0 = torch.ones_like(mask_0).sum(dim=1)
    nnT = torch.ones_like(mask_T).sum(dim=1)

    return_dict = {
        "X_0": X_0_init,
        "X_T": X_T_init,
        "E_0": E_0,
        "E_T": E_T,
        "conn_0": conn_0,
        "conn_T": conn_T,
        "attr_0": attr_0,
        "attr_T": attr_T,
        "ne0": ne0,
        "neT": neT,
        "nn0": nn0,
        "nnT": nnT,
    }
    return return_dict

def construct_K(
    X_0,
    X_T,
    conn_0,
    conn_T,
    attr_0,
    attr_T,
    ne0,
    neT,
    bsz,
    max_num_nodes,
    nn0=None,
    nnT=None,
    dtype=None,
):
    r"""
    X_0, X_T : (bsz, n_node, n_node_feat) # both have the same number of nodes
    conn_0, conn_T : (2, n_edge) # both have different number of edges
    attr_0, attr_T : (n_edge, n_edge_feat)
    ne0 : (n_graphs,) # number of edges for each graph
    neT : (n_graphs,) # number of edges for each graph
    bsz : number of graphs (or batch size)
    max_num_nodes : the number of node in each graph.
    (Optional)
    NOTE: if nn0, nnT is not provided, it is assumed that all graphs have the same number of nodes
    nn0, nnT : (n_graphs,) number of nodes for each graph
    """
    if dtype is None:
        dtype = X_0.dtype

    dev = X_0.device
    X_0 = X_0.reshape(bsz * max_num_nodes, -1)
    X_T = X_T.reshape(bsz * max_num_nodes, -1)

    if nn0 is None or nnT is None:
        nn0 = nnT = torch.ones(bsz, device=dev).long() * max_num_nodes
        batch_0 = torch.arange(bsz, device=dev).repeat_interleave(
            max_num_nodes
        )  # batch index for each node
        batch_T = batch_0  # batch index for each node
    else:
        assert (nn0 == nnT).all()
        batch_0 = torch.arange(bsz, device=dev).repeat_interleave(nn0)
        # batch_T = torch.arange(bsz).repeat_interleave(nnT)

    ptr_0 = torch.cat([torch.LongTensor([0]).to(dev), nn0.cumsum(dim=0)])
    ptr_T = torch.cat([torch.LongTensor([0]).to(dev), nnT.cumsum(dim=0)])

    batch_edge_0 = batch_0[conn_0[0]]
    # batch_edge_T = batch_T[conn_T[0]]
    edge_ptr_0 = torch.cat([torch.LongTensor([0]).to(dev), ne0.cumsum(dim=0)])
    edge_ptr_T = torch.cat([torch.LongTensor([0]).to(dev), neT.cumsum(dim=0)])

    relative_transform = torch.cat(
        [torch.arange(ptr_0[i], ptr_0[i + 1]).to(dev) - ptr_0[i] for i in range(bsz)]
    )

    # construct K_edge
    repeat_edge_index_0 = conn_0.repeat_interleave(neT[batch_edge_0], dim=1)
    a, b = relative_transform[repeat_edge_index_0]
    eT = torch.cat(
        [
            torch.arange(edge_ptr_T[i], edge_ptr_T[i + 1], device=dev).repeat(ne0[i])
            for i in range(bsz)
        ]
    )
    e0 = torch.cat(
        [torch.arange(edge_ptr_0[i], edge_ptr_0[i + 1]).to(dev) for i in range(bsz)]
    ).repeat_interleave(neT[batch_edge_0], dim=0)

    index = torch.stack([eT, a, b])
    # val = (attr_0[e0] * attr_T[eT]).sum(dim=-1)
    val = _lazy_compute(attr_0, attr_T, e0, eT)
    if index.numel() == 0 and val.numel() == 0:
        K_edge = None
    else:
        K_edge = torch.sparse_coo_tensor(
            index, val, size=(neT.sum(), max_num_nodes, max_num_nodes), dtype=dtype
        )

    # construct K_node
    eT = torch.cat(
        [
            torch.arange(ptr_T[i], ptr_T[i + 1], device=dev).repeat_interleave(nn0[i])
            for i in range(bsz)
        ]
    )
    e0 = torch.cat(
        [
            torch.arange(ptr_0[i], ptr_0[i + 1], device=dev).repeat(nnT[i])
            for i in range(bsz)
        ]
    )
    a = relative_transform[e0]

    index = torch.stack([eT, a, a])
    # val = (X_0[e0] * X_T[eT]).sum(dim=-1)
    val = _lazy_compute(X_0, X_T, e0, eT)
    K_node = torch.sparse_coo_tensor(
        index, val, size=(nnT.sum(), max_num_nodes, max_num_nodes), dtype=dtype
    )

    self_loop_index = torch.arange(nnT.sum(), device=dev).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([conn_T, self_loop_index], dim=1)
    if K_edge is not None:
        K = torch.cat([K_edge.to_dense(), K_node.to_dense()], dim=0)
    else:
        K = K_node.to_dense()

    K = K.reshape(-1, max_num_nodes * max_num_nodes)
    return K, edge_index


def _lazy_compute(source1, source2, idx1, idx2, size=1500000):
    r"""
    compute (source1[idx1] * source2[idx2]).sum(dim=-1)
    But, it is lazy in the sense that it does not compute the whole thing at once.
    """
    # print(f"Lazy compute: {source1.size()}, {source2.size()}, {idx1.size()}, {idx2.size()}")
    if idx1.numel() == 0 or idx2.numel() == 0:
        return torch.tensor([], device=source1.device)

    for i in range(0, idx1.size(0), size):
        idx1_ = idx1[i : i + size]
        idx2_ = idx2[i : i + size]
        if i == 0:
            out = (source1[idx1_] * source2[idx2_]).sum(dim=-1)
        else:
            out = torch.cat([out, (source1[idx1_] * source2[idx2_]).sum(dim=-1)], dim=0)
    return out

class MPMModule(MessagePassing):

    def __init__(
        self,
        pooling_type="max",
        max_iter=1000,
        tol=1e-6,
        noise_coeff=1e-6,
        dtype=torch.float32,
    ):
        super(MPMModule, self).__init__(aggr="sum", flow="target_to_source")
        self.pooling_type = pooling_type
        self.max_iter = max_iter
        self.tol = tol
        self.noise_coeff = noise_coeff
        self.dtype = dtype

    def forward(self, x, edge_index, edge_feat):
        return self.propagate(edge_index, x=x, edge_feat=edge_feat)

    def message(self, x_i, x_j, edge_feat):
        n = x_j.shape[-1]
        edge_feat = edge_feat.reshape(-1, n, n)
        if self.pooling_type == "max":
            return (x_j.unsqueeze(1) * edge_feat).max(dim=-1, keepdim=False).values

        elif self.pooling_type == "sum":
            return (x_j.unsqueeze(1) * edge_feat).sum(dim=-1, keepdim=False)

        else:
            raise ValueError(
                "Invalid pooling type, pooling_type should be one of ['max', 'sum']"
            )

    def update(self, aggr_out):
        return aggr_out

    def solve(
        self, K, edge_index, x=None, bsz=None, max_iter=1000, tol=1e-8, noise_coeff=1e-6
    ):
        r"""
        K : cost matrix with shape (num_edges, max_node * batch_size, max_node)
        edge_index : edge index with shape (2, num_edges)
        x : initial matching matrix with shape (max_node * batch_size, max_node)
        """
        K = K.to(self.dtype)
        # assert K.dtype == torch.float64, f"K.dtype : {K.dtype}"
        edge_index = edge_index.to(K.device)

        # init x
        if x is None:
            n = int(K.size(1) ** 0.5)
            if bsz is None:
                raise ValueError("bsz must be provided when x is None")
            x = torch.ones(bsz * n, n).to(K.device).to(K.dtype)

        else:
            n = x.size(1)
            bsz = x.size(0) // n
            assert x.size(0) % n == 0
            x = x.to(K.device).to(K.dtype)

        # noise_coeff = 1e-7
        if noise_coeff > 0:
            K = K + noise_coeff * torch.rand_like(K)

        x = x.reshape(bsz, n**2)
        norm = x.norm(p=2, dim=-1, keepdim=True)
        x = x / norm
        x_last = x.clone()

        for i in range(max_iter):
            x = x.reshape(bsz * n, n)
            x = self.forward(x, edge_index, K)
            x = x.reshape(bsz, n**2)
            norm = x.norm(p=2, dim=-1, keepdim=True)
            x = x / norm

            # print(f"{float((x - x_last).norm(p=2, dim=-1).max()):.2e}")
            if (x - x_last).norm(p=2, dim=-1).max() < tol:
                break
            x_last = x.clone()

        x = x.reshape(bsz, n, n).transpose(1, 2)
        return x


class GraphMatcher(nn.Module):
    def __init__(
        self,
        pooling_type="max",
        max_iter=1000,
        tol=1e-2,
        noise_coeff=1e-6,
        num_seed=5,
        dtype="single",
    ):
        super(GraphMatcher, self).__init__()
        if noise_coeff <= 0:
            warnings.warn("noise_coeff should be non-negative value")
            noise_coeff = 0

        if noise_coeff == 0:
            num_seed = 1

        if num_seed < 1:
            num_seed = 1

        if dtype in ["single", "float32", "float"]:
            self.dtype = torch.float32
        elif dtype in ["double", "float64"]:
            self.dtype = torch.float64
        else:
            raise ValueError(
                "dtype should be one of ['single', 'double', 'float32', 'float64']"
            )

        self.mpm = MPMModule(
            pooling_type=pooling_type,
            max_iter=max_iter,
            tol=tol,
            noise_coeff=noise_coeff,
            dtype=self.dtype,
        )
        self.num_try = num_seed
        self.max_iter = max_iter
        self.noise_coeff = noise_coeff
        self.tol = tol

    def forward(
        self, K, edge_index, n_nodes1=None, n_nodes2=None, bsz=None, local_rank=None
    ):
        X = self.mpm.solve(
            K,
            edge_index,
            bsz=bsz,
            max_iter=self.max_iter,
            tol=self.tol,
            noise_coeff=self.noise_coeff,
        )
        X = pygm.hungarian(X.to("cpu"), n1=n_nodes1, n2=n_nodes2).to(K.device)
        perm = torch.argsort(self.padding(X.argmax(dim=-1), n_nodes1), dim=-1)
        return perm

    def padding(self, X, num_nodes, val=1e6):
        r"""
        pad each row of X at the tail with val
        each row has different size of tail which is defined by num_nodes
        X : (bsz, N)
        num_nodes : (bsz,)
        """
        max_length = X.size(1)
        indices = (
            torch.arange(max_length).expand(len(num_nodes), max_length).to(X.device)
        )
        mask = indices >= num_nodes.unsqueeze(1)

        X = torch.where(mask, torch.full_like(X, val), X)
        return X

    def solve(
        self,
        X_0,
        X_T,
        E_0,
        E_T,
        conn_0,
        conn_T,
        attr_0,
        attr_T,
        ne0,
        neT,
        bsz,
        max_num_nodes,
        nn0=None,
        nnT=None,
        max_iter=None,
        tol=None,
        num_try=None,
        dtype=None,
        local_rank=None,
    ):
        r"""
        solve graph matching problem
        X_0, X_T : (bsz, n_node, n_node_feat) # both have the same number of nodes
        E_0, E_T : (bsz, n_node, n_node, n_edge_feat)
        attr_0, attr_T : (n_edge, n_edge_feat)
        ne0, neT : (n_graphs) # number of edges for each graph
        bsz <class 'int'> : number of graphs (or batch size)
        max_num_nodes <class 'int'> : the number of node in each graph. Each have the same number of nodes
        max_iter <class 'int'> : the maximum number of iteration
        tol <class 'float'> : tolerance for convergence
        num_try <class 'int'> : the number of trials
        """
        if num_try is None:
            num_try = self.num_try
        if dtype is None:
            dtype = self.dtype

        K, edge_index = construct_K(
            X_0,
            X_T,
            conn_0,
            conn_T,
            attr_0,
            attr_T,
            ne0,
            neT,
            bsz,
            max_num_nodes,
            nn0=nn0,
            nnT=nnT,
            dtype=dtype,
        )

        perm_list = []
        for _ in range(num_try):
            perm = self.forward(
                K,
                edge_index,
                n_nodes1=nn0,
                n_nodes2=nnT,
                bsz=bsz,
                local_rank=local_rank,
            )
            perm_list.append(perm)

        perm = self.select_perm(X_0, X_T, E_0, E_T, perm_list)
        nll_init = self.check_nll(X_0, X_T, E_0, E_T)
        X_0, E_0 = self.apply_perm(X_0, E_0, perm)
        nll_final = self.check_nll(X_0, X_T, E_0, E_T)

        if any(nll_init < nll_final):
            warnings.warn(
                "Some graphs are not improved by the graph matching"
                "algorithm. The original graphs are returned."
                f"total : {len(nll_init)}, failure : {torch.sum(nll_init < nll_final)}"
            )
            idx = torch.where(nll_init < nll_final)[0]
            perm[idx] = torch.arange(max_num_nodes).to(perm.device)
            nll_final = torch.min(torch.stack([nll_init, nll_final]), dim=0).values

        return perm, nll_init, nll_final

    def select_perm(self, X_0, X_T, E_0, E_T, perm_list):
        bsz = X_0.size(0)
        nll_list = []
        for perm in perm_list:
            X_0, E_0 = self.apply_perm(X_0, E_0, perm)
            nll = self.check_nll(X_0, X_T, E_0, E_T)
            nll_list.append(nll)

        nll_list = torch.stack(nll_list)  # (num_try, bsz)
        select_index = nll_list.argmin(dim=0)  # (bsz)

        perms = torch.stack(perm_list)
        perms = perms[select_index, torch.arange(bsz)]
        return perms

    def check_nll(self, X_0, X_T, E_0, E_T, eps=1e-18):
        bsz = X_0.size(0)
        nll_X = -torch.log(X_0 + eps) * X_T
        nll_E = -torch.log(E_0 + eps) * E_T
        nll_X = nll_X.reshape(bsz, -1).sum(dim=-1)
        nll_E = nll_E.reshape(bsz, -1).sum(dim=-1)
        nll = nll_X + nll_E
        return nll

    def apply_perm(self, X, E, perm):
        bsz = X.size(0)
        X = X[torch.arange(bsz)[:, None], perm]
        E = E[torch.arange(bsz)[:, None], perm[:, :]].transpose(1, 2)
        E = E[torch.arange(bsz)[:, None], perm[:, :]].transpose(1, 2)
        return X, E
