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
)
from ..metrics.metrics_utils import compute_all_statistics


atom_encoder = {"C": 0, "N": 1, "S": 2, "O": 3, "F": 4, "Cl": 5, "Br": 6}
atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br"]


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class MosesDataset(InMemoryDataset):
    train_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv"
    val_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv"
    test_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv"

    def __init__(
        self,
        split,
        root,
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

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])).float(),
            valencies=load_pickle(self.processed_paths[5]),
        )
        self.smiles = load_pickle(self.processed_paths[6])

    @property
    def raw_file_names(self):
        return ["train_moses.csv", "val_moses.csv", "test_moses.csv"]

    @property
    def split_file_name(self):
        return ["train_moses.csv", "val_moses.csv", "test_moses.csv"]

    @property
    def processed_file_names(self):
        return [
            f"{self.split}.pt",
            f"{self.split}_n.pickle",
            f"{self.split}_node_types.npy",
            f"{self.split}_bond_types.npy",
            f"{self.split}_charge.npy",
            f"{self.split}_valency.pickle",
            f"{self.split}_smiles.pickle",
        ]

    def download(self):
        import rdkit  # noqa

        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, "train_moses.csv"))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, "val_moses.csv"))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, "test_moses.csv"))

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
                unique_charge = set(torch.unique(data.charge).int().numpy())
                charge_list = charge_list.union(unique_charge)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                smiles_kept.append(smile)

        statistics = compute_all_statistics(
            data_list, self.atom_encoder, charge_dic={0: 0}
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
        torch.save(self.collate(data_list), self.processed_paths[0])


class MosesDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.datadir
        base_path = pathlib.Path(get_original_cwd())
        root_path = os.path.join(base_path, self.datadir)

        self.remove_h = False
        datasets = {
            "train": MosesDataset(
                split="train", root=root_path, pre_transform=RemoveYTransform()
            ),
            "val": MosesDataset(
                split="val", root=root_path, pre_transform=RemoveYTransform()
            ),
            "test": MosesDataset(
                split="test", root=root_path, pre_transform=RemoveYTransform()
            ),
        }

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }
        super().__init__(cfg, datasets)


class MosesInfos(AbstractDatasetInfos):
    """
    Moses will not support charge as it only contains one charge type 1
    """

    def __init__(self, datamodule, cfg):
        # basic information
        self.name = "moses"
        self.is_molecular = True
        self.remove_h = False
        self.use_charge = False
        # statistics
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.statistics = datamodule.statistics
        self.collapse_charge = torch.Tensor([-1, 0, 1]).int()
        self.train_smiles = datamodule.train_dataset.smiles
        self.val_smiles = datamodule.val_dataset.smiles
        self.test_smiles = datamodule.test_dataset.smiles
        super().complete_infos(datamodule.statistics, self.atom_encoder)

        # dimensions
        # atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
        self.output_dims = PlaceHolder(X=self.num_node_types, charge=0, E=5, y=0)

        # data specific settings
        # atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
        self.valencies = [4, 3, 2, 2, 1, 1, 1]
        self.atom_weights = [12, 14, 32, 16, 19, 35.4, 79.9]

        self.max_weight = 9 * 80  # Quite arbitrary


def SA_score_data_separation(path, path_greater, path_less):
    RDLogger.DisableLog('rdApp.*')
    list_file = ['train_moses.csv', 'test_moses.csv', 'val_moses.csv']
    os.makedirs(path_greater, exist_ok=True)
    os.makedirs(path_less, exist_ok=True)

    list_file_paths = [os.path.join(path, 'train_moses.csv'),
                os.path.join(path, 'test_moses.csv'),
                os.path.join(path, 'val_moses.csv')]

    all_data = []
    for file in list_file_paths:
        smile_list = pd.read_csv(file)
        smiles_values = smile_list["SMILES"].values.tolist()
        all_data.extend(smiles_values)
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
        file_path = os.path.join(path_less, dataset)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SMILES'])
            for point in selected_less:
                writer.writerow([point])

    for dataset, selected_greater in zip(list_file, greater_lists):
        file_path = os.path.join(path_greater, dataset)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SMILES'])
            for point in selected_greater:
                writer.writerow([point])


if __name__ == "__main__":
    path = './data/moses/moses_pyg/raw'
    path_greater = './data/moses_greater/moses_pyg_greater/raw'
    path_less = './data/moses_less/moses_pyg_less/raw'

    SA_score_data_separation(path, path_greater, path_less)