import pickle
import random
import csv
import sys
import os
import os.path as osp
import pathlib

from rdkit import Chem, DataStructs, RDLogger
print("Found rdkit, all good")
from rdkit.Chem import RDConfig, QED, MolFromSmiles, MolToSmiles
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd

from ..utils import PlaceHolder
from ..datasets.abstract_dataset import (
    MolecularDataModule,
    AbstractDatasetInfos,
)
from ..datasets.dataset_utils import (
    save_pickle,
    mol_to_torch_geometric,
    load_pickle,
    Statistics,
    remove_hydrogens,
    to_list,
    files_exist,
)
from ..metrics.metrics_utils import compute_all_statistics

atom_encoder = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
atom_decoder = [key for key in atom_encoder.keys()]


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data

class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data

class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data

class QM9SmilesDataset(InMemoryDataset):
    def __init__(
        self,
        split,
        root,
        remove_h: bool,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        self.atom_encoder = atom_encoder
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h

        self.atom_encoder = atom_encoder
        if remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])).float(),
            valencies=load_pickle(self.processed_paths[5]),
            real_node_ratio=torch.from_numpy(np.load(self.processed_paths[7])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[6])

    @property
    def raw_file_names(self):
        h = 'noh' if self.remove_h else 'h'
        return [f"train_smiles_qm9_{h}.csv", f"val_smiles_qm9_{h}.csv", f"test_smiles_qm9_{h}.csv"]

    @property
    def split_file_name(self):
        h = 'noh' if self.remove_h else 'h'
        return [f"train_smiles_qm9_{h}.csv", f"val_smiles_qm9_{h}.csv", f"test_smiles_qm9_{h}.csv"]

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        return [
            f"{self.split}_{h}.pt",
            f"{self.split}_n_{h}.pickle",
            f"{self.split}_node_types_{h}.npy",
            f"{self.split}_bond_types_{h}.npy",
            f"{self.split}_charge_{h}.npy",
            f"{self.split}_valency_{h}.pickle",
            f"{self.split}_smiles_{h}.pickle",
            f"{self.split}_real_node_ratio_{h}.npy",
        ]

    def download(self):
        pass

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        smile_list = pd.read_csv(self.raw_paths[self.file_idx])
        smile_list = smile_list["SMILES"].values
        data_list = []
        smiles_kept = []
        charge_list = set()

        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)

            if mol is not None:
                data = mol_to_torch_geometric(mol, atom_encoder, smile)
                if self.remove_h:
                    data = remove_hydrogens(data)
                unique_charge = set(torch.unique(data.charge).int().numpy())
                charge_list = charge_list.union(unique_charge)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                if data.edge_index.numel() > 0:
                    data_list.append(data)

                smiles_kept.append(smile)

        statistics = compute_all_statistics(
            data_list, self.atom_encoder, charge_dic={-1: 0, 0: 1, 1: 2}
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.node_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        print(
            "Number of molecules that could not be mapped to smiles: ",
            len(smile_list) - len(smiles_kept),
        )
        save_pickle(set(smiles_kept), self.processed_paths[6])

        for data in data_list:
            data.x = F.one_hot(data.x.to(torch.long), num_classes=len(statistics.node_types)).to(torch.float)
            data.edge_attr = F.one_hot(data.edge_attr.to(torch.long), num_classes=len(statistics.bond_types)).to(torch.float)
            data.charge = F.one_hot(data.charge.to(torch.long) + 1, num_classes=len(statistics.charge_types[0])).to(torch.float)
        torch.save(self.collate(data_list), self.processed_paths[0])
        np.save(self.processed_paths[7], statistics.real_node_ratio)


class QM9SmilesDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.datadir
        base_path = pathlib.Path(get_original_cwd())
        root_path = os.path.join(base_path, self.datadir)

        self.remove_h = cfg.remove_h
        datasets = {
            "train": QM9SmilesDataset(
                split="train", root=root_path, remove_h=self.cfg.remove_h, pre_transform=RemoveYTransform()
            ),
            "val": QM9SmilesDataset(
                split="val", root=root_path, remove_h=self.cfg.remove_h, pre_transform=RemoveYTransform()
            ),
            "test": QM9SmilesDataset(
                split="test", root=root_path, remove_h=self.cfg.remove_h, pre_transform=RemoveYTransform()
            ),
        }

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }
        super().__init__(cfg, datasets)
        super().prepare_dataloader()
        self.inner = self.train_dataset
        self.testing = self.test_dataset
        self.validating = self.val_dataset


class QM9SmilesInfos(AbstractDatasetInfos):
    """
    Moses will not support charge as it only contains one charge type 1
    """

    def __init__(self, datamodule, cfg):
        # basic information
        self.name = "qm9_smiles"
        self.is_molecular = True
        self.remove_h = cfg.remove_h
        self.use_charge = cfg.use_charge
        self.collapse_charges = torch.Tensor([-1, 0, 1]).int()
        # statistics
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.statistics = datamodule.statistics
        self.collapse_charge = torch.Tensor([-1, 0, 1]).int()
        self.train_smiles = datamodule.train_dataset.smiles
        self.val_smiles = datamodule.val_dataset.smiles
        self.test_smiles = datamodule.test_dataset.smiles
        if self.remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }
            self.atom_decoder = [key for key in self.atom_encoder.keys()]
        super().complete_infos(datamodule.statistics, self.atom_encoder)

        # dimensions
        self.output_dims = PlaceHolder(X=self.num_node_types, charge=self.num_charge_types, E=5, y=0)
        if not self.use_charge:
            self.output_dims.charge = 0

        # data specific settings
        self.valencies = [4, 3, 2, 1] if self.remove_h else [1, 4, 3, 2, 1]
        self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19} if self.remove_h else {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
        self.max_weight = 40 * 19  # Quite arbitrary

        if self.remove_h:
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.tensor(
                [2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073]
            )
        else:
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.tensor(
                [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
            )


def SA_score_data_separation(path, path_greater, path_greater_more, path_greater_less, path_less, remove_h):
    RDLogger.DisableLog('rdApp.*')
    h = 'noh' if remove_h else 'h'
    list_file = [f'train_smiles_qm9_{h}', f'test_smiles_qm9_{h}', f'val_smiles_qm9_{h}']
    os.makedirs(path_greater, exist_ok=True)
    os.makedirs(path_greater_more, exist_ok=True)
    os.makedirs(path_greater_less, exist_ok=True)
    os.makedirs(path_less, exist_ok=True)

    all_data = []
    for dataset in list_file:
        with open(os.path.join(path, dataset + '.pickle'), 'rb') as file:
            data = pickle.load(file)
            all_data.extend(data)
    random.shuffle(all_data)

    sa_greater_3 = []
    sa_less_3 = []

    for smiles in all_data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            sa_score = sascorer.calculateScore(mol)
            if sa_score <= 3:
                sa_less_3.append(smiles)
            elif sa_score > 3:
                sa_greater_3.append(smiles)

    # Split sizes for the SA <= 3 (favourable and smaller dataset)
    less_3_size = len(sa_less_3)
    split1_less = int(less_3_size * 0.8)
    split2_less = int(less_3_size * 0.9)

    train_less = sa_less_3[:split1_less]
    val_less = sa_less_3[split1_less:split2_less]
    test_less = sa_less_3[split2_less:]

    train_greater = sa_greater_3[:split1_less]
    val_greater = sa_greater_3[split1_less:split2_less]
    test_greater = sa_greater_3[split2_less:]

    less_lists = [train_less, test_less, val_less]
    greater_lists = [train_greater, test_greater, val_greater]

    for dataset, selected_less in zip(list_file, less_lists):
        file_path = os.path.join(path_less, dataset + '.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SMILES'])
            for point in selected_less:
                writer.writerow([point])

    for dataset, selected_greater in zip(list_file, greater_lists):
        file_path = os.path.join(path_greater, dataset + '.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SMILES'])
            for point in selected_greater:
                writer.writerow([point])

    ecart = int(2 * split1_less)
    cur_split1_less = split1_less + ecart
    cur_split2_less = split2_less + ecart
    train_greater = sa_greater_3[:cur_split1_less]
    val_greater = sa_greater_3[cur_split1_less:cur_split2_less]
    test_greater = sa_greater_3[cur_split2_less:]

    greater_lists = [train_greater, test_greater, val_greater]

    for dataset, selected_greater in zip(list_file, greater_lists):
        file_path = os.path.join(path_greater_more, dataset + '.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SMILES'])
            for point in selected_greater:
                writer.writerow([point])

    ecart = int(0.5 * split1_less)
    cur_split1_less = split1_less - ecart
    cur_split2_less = split2_less - ecart
    train_greater = sa_greater_3[:cur_split1_less]
    val_greater = sa_greater_3[cur_split1_less:cur_split2_less]
    test_greater = sa_greater_3[cur_split2_less:]

    greater_lists = [train_greater, test_greater, val_greater]

    for dataset, selected_greater in zip(list_file, greater_lists):
        file_path = os.path.join(path_greater_less, dataset + '.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SMILES'])
            for point in selected_greater:
                writer.writerow([point])



if __name__ == "__main__":
    path = './data/qm9/qm9_pyg/processed'
    path_greater = './data/qm9_greater/qm9_pyg_greater/raw'
    path_greater_more = './data/qm9_greater/qm9_pyg_greater_more/raw'
    path_greater_less = './data/qm9_greater/qm9_pyg_greater_less/raw'
    path_less = './data/qm9_less/qm9_pyg_less/raw'
    remove_h = True

    SA_score_data_separation(path, path_greater, path_greater_more, path_greater_less, path_less, remove_h)