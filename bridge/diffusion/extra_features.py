import math

import torch
from .. import utils


class DummyExtraFeatures:
    def __init__(self):
        """This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data.X
        E = noisy_data.E
        y = noisy_data.y
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class PositionalEncoding:
    def __init__(self, n_max_dataset, D=30):
        self.n_max = n_max_dataset
        self.d = math.floor(D / 2)

    def __call__(self, dense_noisy_data):
        device = dense_noisy_data.X.device
        n_max_batch = dense_noisy_data.X.shape[1]

        arange_n = torch.arange(n_max_batch, device=device)  # n_max
        arange_d = torch.arange(self.d, device=device)  # d
        frequencies = math.pi / torch.pow(
            self.n_max, 2 * arange_d / self.d
        )  # d

        sines = torch.sin(
            arange_n.unsqueeze(1) * frequencies.unsqueeze(0)
        )  # N, d
        cosines = torch.cos(
            arange_n.unsqueeze(1) * frequencies.unsqueeze(0)
        )  # N, d
        encoding = torch.hstack((sines, cosines))  # N, D
        extra_x = encoding.unsqueeze(0)  # 1, N, D
        extra_x = extra_x * dense_noisy_data.node_mask.unsqueeze(-1)  # B, N, D
        return extra_x


class ExtraFeatures:
    def __init__(
        self,
        extra_features_type,
        dataset_info,
        rrwp_steps,
        use_positional: bool,
    ):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        self.rrwp_steps = rrwp_steps

        self.use_positional = use_positional
        if use_positional:
            self.positional_encoding = PositionalEncoding(
                dataset_info.max_n_nodes
            )

        self.RRWP = RRWPFeatures()
        if extra_features_type in ["eigenvalues", "all"]:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def __call__(self, noisy_data):
        n = noisy_data.node_mask.sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(noisy_data)  # (bs, n_cycles)

        if self.features_type == "cycles":
            E = noisy_data.E
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            feats = utils.PlaceHolder(
                X=x_cycles,
                E=extra_edge_attr,
                y=torch.hstack((n, y_cycles)),
            )

        elif self.features_type == "rrwp":
            E = noisy_data.E.float()[..., 1:].sum(-1)  # bs, n, n
            rrwp_edge_attr = self.RRWP(E, k=self.rrwp_steps)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            feats - utils.PlaceHolder(
                X=rrwp_node_attr,
                E=rrwp_edge_attr,
                y=torch.hstack((n, y_cycles)),
            )

        elif self.features_type == "rrwp_comp":
            E = noisy_data.E.float()[..., 1:].sum(-1)  # bs, n, n
            rrwp_edge_attr = self.RRWP(E, k=int(self.rrwp_steps / 2))
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            comp_E = 1 - noisy_data.E.float()[..., 1:].sum(-1)  # bs, n, n
            comp_rrwp_edge_attr = self.RRWP(comp_E, k=int(self.rrwp_steps / 2))
            comp_rrwp_node_attr = comp_rrwp_edge_attr[
                :, diag_index, diag_index, :
            ]

            feats = utils.PlaceHolder(
                X=torch.cat(
                    (rrwp_node_attr, comp_rrwp_node_attr),
                    dim=-1,
                ),
                E=torch.cat(
                    (rrwp_edge_attr, comp_rrwp_edge_attr),
                    dim=-1,
                ),
                y=torch.hstack((n, y_cycles)),
            )

        elif self.features_type == "all":
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data.E
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            (
                n_components,
                batched_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigvec,
            ) = eigenfeatures  # (bs, 1), (bs, 10),

            feats = utils.PlaceHolder(
                X=torch.cat(
                    (
                        x_cycles,
                        nonlcc_indicator,
                        k_lowest_eigvec,
                    ),
                    dim=-1,
                ),
                E=extra_edge_attr,
                y=torch.hstack(
                    (
                        n,
                        y_cycles,
                        n_components,
                        batched_eigenvalues,
                    )
                ),
            )

            # rrwp_edge_attr = self.RRWP(noisy_data)
            # diag_index = torch.arange(extra_edge_attr.shape[1])
            # rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

            # return utils.PlaceHolder(
            #     X=torch.cat(
            #         (x_cycles, nonlcc_indicator, k_lowest_eigvec, rrwp_node_attr),
            #         dim=-1,
            #     ),
            #     E=torch.cat((extra_edge_attr, rrwp_edge_attr), dim=-1),
            #     y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)),
            # )
        else:
            raise ValueError(
                f"Features type {self.features_type} not implemented"
            )

        if self.use_positional:
            node_feat = self.positional_encoding(noisy_data)
            feats.X = torch.cat([feats.X, node_feat], dim=-1)

        return feats


class RRWPFeatures:
    def __init__(self, k=10):
        self.k = k

    def __call__(self, E, k):
        # degree = E.sum(dim=-1).float().unsqueeze(1)  # bs, 1, n
        # degree = degree.repeat(1, E.shape[1], 1)  # bs, n, n
        (
            bs,
            n,
            _,
        ) = E.shape
        degree = torch.zeros(bs, n, n, device=E.device)
        to_fill = 1 / (E.sum(dim=-1).float())
        to_fill[E.sum(dim=-1).float() == 0] = 0
        degree = torch.diagonal_scatter(degree, to_fill, dim1=1, dim2=2)
        E = degree @ E

        rrwp_list = [E]
        # multiplication_list = [E]
        for i in range(k):
            cur_rrwp = rrwp_list[-1] @ E
            rrwp_list.append(cur_rrwp)
            # cur_rrwp = E @ multiplication_list[-1]
            # multiplication_list.append(cur_rrwp)
            # rrwp_top = torch.triu(cur_rrwp, diagonal=1)
            # rrwp_bot = torch.tril(cur_rrwp, diagonal=1)
            # rrwp_top = rrwp_top + rrwp_top.transpose(1, 2)
            # rrwp_bot = rrwp_bot + rrwp_bot.transpose(1, 2)
            # rrwp_list.append(rrwp_top)
            # rrwp_list.append(rrwp_bot)

        return torch.stack(rrwp_list, -1)


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        adj_matrix = noisy_data.E[..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(
            adj_matrix=adj_matrix
        )  # (bs, n_cycles)
        x_cycles = x_cycles.type_as(
            adj_matrix
        ) * noisy_data.node_mask.unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """

    def __init__(self, mode):
        """mode: 'eigenvalues' or 'all'"""
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data.E
        mask = noisy_data.node_mask
        A = (
            E_t[..., 1:].sum(dim=-1).float()
            * mask.unsqueeze(1)
            * mask.unsqueeze(2)
        )
        # L = compute_laplacian(A, normalize="sym")
        L = compute_laplacian(A, normalize=False)
        mask_diag = (
            2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        )
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == "eigenvalues":
            eigvals = torch.linalg.eigvalsh(L)  # bs, n
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
                eigenvalues=eigvals
            )
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == "all":
            eigvals, eigvectors = torch.linalg.eigh(L)
            # print(eigvals)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
                eigenvalues=eigvals
            )

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
                vectors=eigvectors,
                node_mask=noisy_data.node_mask,
                n_connected=n_connected_comp,
            )
            return (
                n_connected_comp,
                batch_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigenvector,
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)  # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)  # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency  # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)  # (bs, n)
    D_norm = torch.diag_embed(diag_norm)  # (bs, n, n)
    L = (
        torch.eye(n, device=adjacency.device).unsqueeze(0)
        - D_norm @ adjacency @ D_norm
    )
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    try:
        assert (n_connected_components > 0).all(), (
            n_connected_components,
            ev,
        )
    except:
        import pdb

        pdb.set_trace()

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack(
            (
                eigenvalues,
                2 * torch.ones(bs, to_extend).type_as(eigenvalues),
            )
        )
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(
        0
    ) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask  # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (
        ~node_mask
    )  # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(
        first_ev, dim=1
    ).values  # values: bs -- indices: bs
    mask = ~(first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat(
            (
                vectors,
                torch.zeros(bs, n, to_extend).type_as(vectors),
            ),
            dim=2,
        )  # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(
        0
    ) + n_connected.unsqueeze(
        2
    )  # bs, 1, k
    indices = indices.expand(-1, n, -1)  # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """Builds cycle counts for each node in a graph."""

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """tr(A ** 3)."""
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (
            torch.sum(c3, dim=-1) / 6
        ).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        )
        return (c4 / 2).unsqueeze(-1).float(), (
            torch.sum(c4, dim=-1) / 8
        ).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = (
            diag_a5
            - 2 * triangles * self.d
            - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1)
            + triangles
        )
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(
            -1
        ).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix**2)
        term3_t = torch.sum(
            self.adj_matrix * self.k2_matrix.pow(2),
            dim=[-2, -1],
        )
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
