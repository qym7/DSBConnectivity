import os
from collections import Counter
import pandas as pd
import csv

# from fcd import get_fcd, load_ref_model
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

# from moses.metrics.metrics import get_all_metrics
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric as pyg
import torch_geometric.nn.pool as pool
from torchmetrics import (
    Metric,
    MetricCollection,
    MeanAbsoluteError,
)

from rdkit import Chem
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# from ..metrics.molecular_metrics import Molecule
from .. import utils
from ..analysis.rdkit_functions import compute_molecular_metrics


allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {
    6: 4,
    7: 3,
    8: 2,
    9: 1,
    15: 3,
    16: 2,
    17: 1,
    35: 1,
    53: 1,
}


class CEPerClass(Metric):
    full_state_update = False

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


class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {
            "H": HydrogenCE,
            "C": CarbonCE,
            "N": NitroCE,
            "O": OxyCE,
            "F": FluorCE,
            "B": BoronCE,
            "Br": BrCE,
            "Cl": ClCE,
            "I": IodineCE,
            "P": PhosphorusCE,
            "S": SulfurCE,
            "Se": SeCE,
            "Si": SiCE,
        }

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


class TrainMolecularMetricsDiscrete(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(
        self,
        masked_pred_X,
        masked_pred_E,
        true_X,
        true_E,
        log: bool,
    ):
        self.train_atom_metrics(masked_pred_X, true_X)
        self.train_bond_metrics(masked_pred_E, true_E)
        if log:
            to_log = {}
            for (
                key,
                val,
            ) in self.train_atom_metrics.compute().items():
                to_log["train/" + key] = val.item()
            for (
                key,
                val,
            ) in self.train_bond_metrics.compute().items():
                to_log["train/" + key] = val.item()
            if wandb.run:
                wandb.log(to_log, commit=False)

    def reset(self):
        for metric in [
            self.train_atom_metrics,
            self.train_bond_metrics,
        ]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log["train_epoch/" + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log["train_epoch/" + key] = val.item()
        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = val.item()
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = val.item()

        return epoch_atom_metrics, epoch_bond_metrics


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, test_smiles, train_smiles):
        super().__init__()
        di = dataset_infos

        self.dataset_infos = di
        self.generated_n_dist = GeneratedNDistribution(
            di.max_n_nodes
        )  # this metrics is essential when using virtual nodes
        self.generated_node_dist = GeneratedNodesDistribution(di.num_node_types)
        self.generated_edge_dist = GeneratedEdgesDistribution(di.num_edge_types)
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)

        n_target_dist = di.n_nodes.type_as(self.generated_n_dist.n_dist)
        n_target_dist = n_target_dist / torch.sum(n_target_dist)
        self.register_buffer("n_target_dist", n_target_dist)

        node_target_dist = di.node_types.type_as(self.generated_node_dist.node_dist)
        node_target_dist = node_target_dist / torch.sum(node_target_dist)
        self.register_buffer("node_target_dist", node_target_dist)

        edge_target_dist = di.edge_types.type_as(self.generated_edge_dist.edge_dist)
        edge_target_dist = edge_target_dist / torch.sum(edge_target_dist)
        self.register_buffer("edge_target_dist", edge_target_dist)

        valency_target_dist = di.valency_distribution.type_as(
            self.generated_valency_dist.edgepernode_dist
        )
        valency_target_dist = valency_target_dist / torch.sum(valency_target_dist)
        self.register_buffer("valency_target_dist", valency_target_dist)

        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = train_smiles
        self.test_smiles = test_smiles
        self.dataset_info = di

    def forward(
        self,
        graphs: list,
        current_epoch,
        local_rank,
        fb,
        test=False,
        source_graphs=None,
    ):
        molecules = graphs
        (
            stability,
            rdkit_metrics,
            all_smiles,
            to_log,
            sa_values_tuple,
        ) = compute_molecular_metrics(
            molecules,
            self.test_smiles,
            self.train_smiles,
            self.dataset_info,
            fb,
            source_graphs,
        )

        if test and sa_values_tuple[0] is not None:
            sa_data = {
                "source_smiles": all_smiles[0],
                "target_smiles": all_smiles[1],
                "source_sa": sa_values_tuple[0],
                "target_sa": sa_values_tuple[1],
            }
            df = pd.DataFrame(sa_data)
            df.to_csv("SA_values.csv", index=False)

        all_smiles = all_smiles[1]
        if test and local_rank == 0:
            with open(r"final_smiles.csv", "w", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["SMILES"])
                for smiles in all_smiles:
                    writer.writerow([smiles])
                print("All smiles saved")

        print("Starting custom metrics")
        self.generated_n_dist(molecules)
        generated_n_dist = self.generated_n_dist.compute()
        self.n_dist_mae(generated_n_dist)

        self.generated_node_dist(molecules)
        generated_node_dist = self.generated_node_dist.compute()
        self.node_dist_mae(generated_node_dist)

        self.generated_edge_dist(molecules)
        generated_edge_dist = self.generated_edge_dist.compute()
        self.edge_dist_mae(generated_edge_dist)

        self.generated_valency_dist(molecules)
        generated_valency_dist = self.generated_valency_dist.compute()
        self.valency_dist_mae(generated_valency_dist)

        for i, atom_type in enumerate(self.dataset_info.atom_decoder):
            generated_probability = generated_node_dist[i]
            target_probability = self.node_target_dist[i]
            to_log[f"molecular_metrics/{atom_type}_dist"] = (
                generated_probability - target_probability
            ).item()

        for j, bond_type in enumerate(
            [
                "No bond",
                "Single",
                "Double",
                "Triple",
                "Aromatic",
            ]
        ):
            generated_probability = generated_edge_dist[j]
            target_probability = self.edge_target_dist[j]
            to_log[f"molecular_metrics/bond_{bond_type}_dist"] = (
                generated_probability - target_probability
            ).item()

        for valency in range(6):
            generated_probability = generated_valency_dist[valency]
            target_probability = self.valency_target_dist[valency]
            to_log[f"molecular_metrics/valency_{valency}_dist"] = (
                generated_probability - target_probability
            ).item()

        n_mae = self.n_dist_mae.compute()
        node_mae = self.node_dist_mae.compute()
        edge_mae = self.edge_dist_mae.compute()
        valency_mae = self.valency_dist_mae.compute()

        to_log.update({"basic_metrics/n_mae": n_mae.item()})
        to_log.update({"basic_metrics/node_mae": node_mae.item()})
        to_log.update({"basic_metrics/edge_mae": edge_mae.item()})
        to_log.update({"basic_metrics/valency_mae": valency_mae.item()})

        if wandb.run:
            wandb.log(to_log, commit=False)
            wandb.run.summary["Gen n distribution"] = generated_n_dist
            wandb.run.summary["Gen node distribution"] = generated_node_dist
            wandb.run.summary["Gen edge distribution"] = generated_edge_dist
            wandb.run.summary["Gen valency distribution"] = generated_valency_dist

        if local_rank == 0:
            print("Custom metrics computed.")

        return to_log

    def reset(self):
        for metric in [
            self.n_dist_mae,
            self.node_dist_mae,
            self.edge_dist_mae,
            self.valency_dist_mae,
        ]:
            metric.reset()


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class MeanNumberEdge(Metric):
    full_state_update = False

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


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)


class MSEPerClass(MeanSquaredError):
    full_state_update = False

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds = preds[..., self.class_id]
        target = target[..., self.class_id]
        super().update(preds, target)


class HydroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


# Bonds MSE


class NoBondMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetrics(MetricCollection):
    def __init__(self, dataset_infos):
        remove_h = dataset_infos.remove_h
        self.atom_decoder = dataset_infos.atom_decoder
        num_atom_types = len(self.atom_decoder)

        types = {
            "H": 0,
            "C": 1,
            "N": 2,
            "O": 3,
            "F": 4,
            "B": 5,
            "Br": 6,
            "Cl": 7,
            "I": 8,
            "P": 9,
            "S": 10,
            "Se": 11,
            "Si": 12,
        }

        class_dict = {
            "H": HydroMSE,
            "C": CarbonMSE,
            "N": NitroMSE,
            "O": OxyMSE,
            "F": FluorMSE,
            "B": BoronMSE,
            "Br": BrMSE,
            "Cl": ClMSE,
            "I": IodineMSE,
            "P": PhosphorusMSE,
            "S": SulfurMSE,
            "Se": SeMSE,
            "Si": SiMSE,
        }

        metrics_list = []
        for i, atom_type in enumerate(self.atom_decoder):
            metrics_list.append(class_dict[atom_type](i))

        super().__init__(metrics_list)


class BondMetrics(MetricCollection):
    def __init__(self):
        mse_no_bond = NoBondMSE(0)
        mse_SI = SingleMSE(1)
        mse_DO = DoubleMSE(2)
        mse_TR = TripleMSE(3)
        mse_AR = AromaticMSE(4)
        super().__init__([mse_no_bond, mse_SI, mse_DO, mse_TR, mse_AR])


if __name__ == "__main__":
    from torchmetrics.utilities import (
        check_forward_full_state_property,
    )


class Molecule:
    def __init__(
        self,
        node_types: Tensor,
        bond_types: Tensor,
        atom_decoder,
        charge,
    ):
        """node_types: n      LongTensor
        charge: n         LongTensor
        bond_types: n x n  LongTensor
        atom_decoder: extracted from dataset_infos."""

        assert node_types.dim() == 1 and node_types.dtype == torch.long, (
            f"shape of atoms {node_types.shape} " f"and dtype {node_types.dtype}"
        )
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, (
            f"shape of bonds {bond_types.shape} --" f" {bond_types.dtype}"
        )
        assert len(node_types.shape) == 1
        assert len(bond_types.shape) == 2

        self.node_types = node_types.long()
        self.bond_types = bond_types.long()
        self.charge = charge if charge is not None else torch.zeros_like(node_types)
        self.charge = self.charge.long()
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.atom_decoder = atom_decoder
        self.num_nodes = len(node_types)
        self.num_node_types = len(atom_decoder)
        self.device = self.node_types.device

    def build_molecule(self, atom_decoder):
        """If positions is None,"""
        mol = Chem.RWMol()
        for atom, charge in zip(self.node_types, self.charge):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            if charge.numel() > 0:
                a.SetFormalCharge(charge.item())
            mol.AddAtom(a)

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        return mol

    def check_stability(self, debug=False):
        e = self.bond_types.clone()
        e[e == 4] = 1.5
        e[e < 0] = 0
        valencies = torch.sum(e, dim=-1).long()

        n_stable_at = 0
        mol_stable = True
        for i, (atom_type, valency, charge) in enumerate(
            zip(self.node_types, valencies, self.charge)
        ):
            atom_type = atom_type.item()
            valency = valency.item()
            charge = charge.item()
            possible_bonds = allowed_bonds[self.atom_decoder[atom_type]]
            if type(possible_bonds) == int:
                is_stable = possible_bonds == valency
            elif type(possible_bonds) == dict:
                expected_bonds = (
                    possible_bonds[charge]
                    if charge in possible_bonds.keys()
                    else possible_bonds[0]
                )
                is_stable = (
                    expected_bonds == valency
                    if type(expected_bonds) == int
                    else valency in expected_bonds
                )
            else:
                is_stable = valency in possible_bonds
            if not is_stable:
                mol_stable = False
            if not is_stable and debug:
                print(
                    f"Invalid atom {self.atom_decoder[atom_type]}: valency={valency}, charge={charge}"
                )
                print()
            n_stable_at += int(is_stable)

        return mol_stable, n_stable_at, len(self.node_types)
