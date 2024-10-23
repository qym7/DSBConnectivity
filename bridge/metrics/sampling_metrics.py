from collections import Counter
import math

import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import wandb
import torch
from torch import Tensor
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric as pyg
from torchmetrics import (
    MeanMetric,
    MaxMetric,
    Metric,
    MeanAbsoluteError,
)

from ..utils import undirected_to_directed
from ..metrics.metrics_utils import (
    counter_to_tensor,
    wasserstein1d,
    total_variation1d,
)


class SamplingMetrics(nn.Module):
    def __init__(self, dataset_infos, test, dataloaders=None):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.test = test

        self.disconnected = MeanMetric()
        self.mean_components = MeanMetric()
        self.max_components = MaxMetric()
        self.num_nodes_w1 = MeanMetric()
        self.node_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()

        self.domain_metrics = None
        if dataset_infos.is_molecular:
            from ..metrics.molecular_metrics import (
                SamplingMolecularMetrics,
            )

            all_val_smiles = set(
                list(dataset_infos.val_smiles)
                + list(dataset_infos.test_smiles)
            )
            self.domain_metrics = SamplingMolecularMetrics(
                dataset_infos,
                # dataset_infos.val_smiles if not test else dataset_infos.test_smiles,
                all_val_smiles,
                dataset_infos.train_smiles,
            )

        elif dataset_infos.spectre:
            from ..metrics.spectre_utils import (
                Comm20SamplingMetrics,
                PlanarSamplingMetrics,
                SBMSamplingMetrics,
                ProteinSamplingMetrics,
                EgoSamplingMetrics,
            )

            # self.domain_metrics = Comm20SamplingMetrics(dataloaders=dataloaders)
            if dataset_infos.dataset_name == "comm20":
                self.domain_metrics = Comm20SamplingMetrics(
                    dataloaders=dataloaders
                )
            elif "planar" in dataset_infos.dataset_name:
                self.domain_metrics = PlanarSamplingMetrics(
                    dataloaders=dataloaders
                )
            elif "sbm" in dataset_infos.dataset_name:
                self.domain_metrics = SBMSamplingMetrics(
                    dataloaders=dataloaders
                )
            elif dataset_infos.dataset_name == "protein":
                self.domain_metrics = ProteinSamplingMetrics(
                    dataloaders=dataloaders
                )
            elif dataset_infos.dataset_name == "ego":
                self.domain_metrics = EgoSamplingMetrics(
                    dataloaders=dataloaders
                )
            else:
                raise ValueError(
                    "Dataset {} not implemented".format(
                        dataset_infos.dataset_name
                    )
                )

    def reset(self):
        for metric in [
            self.mean_components,
            self.max_components,
            self.disconnected,
            self.num_nodes_w1,
            self.node_types_tv,
            self.edge_types_tv,
        ]:
            metric.reset()
        if self.domain_metrics is not None:
            self.domain_metrics.reset()

    def compute_all_metrics(
        self,
        generated_graphs: list,
        current_epoch,
        local_rank,
        fb,
        i,
        source_graphs,
    ):
        """Compare statistics of the generated data with statistics of the val/test set"""
        self.reset()
        key = f"val_{fb}" if not self.test else f"test_{fb}"
        to_log = {
            f"{key}/NumNodesW1": self.num_nodes_w1.compute().item(),
            f"{key}/NodeTypesTV": self.node_types_tv.compute().item(),
            f"{key}/EdgeTypesTV": self.edge_types_tv.compute().item(),
            f"{key}/Disconnected": self.disconnected.compute().item() * 100,
            f"{key}/MeanComponents": self.mean_components.compute().item(),
            f"{key}/MaxComponents": self.max_components.compute().item(),
        }

        for k in to_log:
            if math.isnan(to_log[k]):
                to_log[k] = 0.0
            if math.isinf(to_log[k]):
                to_log[k] = 0.0

        if self.domain_metrics is not None:
            if self.dataset_infos.is_molecular:
                domain_key = (
                    f"domain_val_{fb}"
                    if not self.test
                    else f"domain_test_{fb}"
                )
                do_metrics = self.domain_metrics.forward(
                    generated_graphs,
                    current_epoch,
                    local_rank,
                    fb,
                    test=self.test,
                    source_graphs=source_graphs,
                )
                do_metrics = {
                    f"{domain_key}/{k}": do_metrics[k] for k in do_metrics
                }
                to_log.update(do_metrics)
            else:
                domain_key = (
                    f"domain_val_{fb}"
                    if not self.test
                    else f"domain_test_{fb}"
                )
                do_metrics = self.domain_metrics.forward(
                    generated_graphs,
                    current_epoch,
                    local_rank,
                    test=self.test,
                )
                do_metrics = {
                    f"{domain_key}/{k}": do_metrics[k] for k in do_metrics
                }
                to_log.update(do_metrics)

        # if wandb.run:
        #     wandb.log(to_log, commit=False)
        to_log = {k: float(to_log[k]) for k in to_log}
        print(to_log)

        return to_log


def number_nodes_distance(generated_graphs, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(
        max_number_nodes + 1,
        device=generated_graphs.batch.device,
    )
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for i in range(generated_graphs.batch.max() + 1):
        c[int((generated_graphs.batch == i).sum())] += 1

    generated_n = counter_to_tensor(c).to(reference_n.device)
    return wasserstein1d(generated_n, reference_n)


def node_types_distance(generated_graphs, target, save_histogram=True):
    generated_distribution = torch.zeros_like(target)

    for node in generated_graphs.node:
        generated_distribution[node] += 1

    if save_histogram:
        if wandb.run:
            data = [
                [k, l]
                for k, l in zip(
                    target,
                    generated_distribution / generated_distribution.sum(),
                )
            ]
            table = wandb.Table(data=data, columns=["target", "generate"])
            wandb.log(
                {
                    "node distribution": wandb.plot.histogram(
                        table,
                        "types",
                        title="node distribution",
                    )
                }
            )

        np.save(
            "generated_node_types.npy",
            generated_distribution.cpu().numpy(),
        )

    return total_variation1d(generated_distribution, target)


def bond_types_distance(generated_graphs, target, save_histogram=True):
    device = generated_graphs.batch.device
    generated_distribution = torch.zeros_like(target).to(device)
    edge_index, edge_attr = undirected_to_directed(
        generated_graphs.edge_index, generated_graphs.edge_attr
    )
    for edge in edge_attr:
        generated_distribution[edge] += 1

    # get the number of non-existing edges
    n_nodes = pyg.nn.pool.global_add_pool(
        torch.ones_like(generated_graphs.batch).unsqueeze(-1),
        generated_graphs.batch,
    ).flatten()
    generated_distribution[0] = (n_nodes * (n_nodes - 1) / 2).sum()
    generated_distribution[0] = (
        generated_distribution[0] - generated_distribution[1:].sum()
    )

    if save_histogram:
        if wandb.run:
            data = [
                [k, l]
                for k, l in zip(
                    target,
                    generated_distribution / generated_distribution.sum(),
                )
            ]
            table = wandb.Table(data=data, columns=["target", "generate"])
            wandb.log(
                {
                    "edge distribution": wandb.plot.histogram(
                        table,
                        "types",
                        title="edge distribution",
                    )
                }
            )

        np.save(
            "generated_bond_types.npy",
            generated_distribution.cpu().numpy(),
        )

    tv, tv_per_class = total_variation1d(
        generated_distribution, target.to(device)
    )
    return tv, tv_per_class


def connected_components(generated_graphs):
    num_graphs = int(generated_graphs.batch.max() + 1)
    all_num_components = torch.zeros(num_graphs)
    batch = generated_graphs.batch
    edge_batch = batch[generated_graphs.edge_index[0]]
    for i in range(num_graphs):
        # get the graph
        node_mask = batch == i
        edge_mask = edge_batch == i
        node = generated_graphs.node[node_mask]
        edge_index = (
            generated_graphs.edge_index[:, edge_mask] - generated_graphs.ptr[i]
        )
        # DENSE OPERATIONS
        sp_adj = to_scipy_sparse_matrix(edge_index, num_nodes=len(node))
        num_components, component = sp.csgraph.connected_components(
            sp_adj.toarray()
        )
        all_num_components[i] = num_components

    return all_num_components


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state(
            "total_ce",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.0).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state(
            "total_edge",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_samples",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples
