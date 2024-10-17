import os
import os.path as osp
import pathlib


import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
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


atom_encoder = {"C": 0, "F": 1, "S": 2, "I": 3, "Cl": 4, "P": 5, "Br": 6, "O": 7, "N": 8}
atom_decoder = ['C', 'F', 'S', 'I', 'Cl', 'P', 'Br', 'O', 'N']


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class ZincDataset(InMemoryDataset):
    train_url = 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/refs/heads/main/models/zinc/250k_rndm_zinc_drugs_clean_3.csv'

    def __init__(
        self,
        split,
        root,
        transfer,
        is_target,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        self.transfer = transfer
        self.atom_encoder = atom_encoder
        self.is_target = is_target
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
            real_node_ratio=torch.from_numpy(np.load(self.processed_paths[7])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[6])

    @property
    def raw_file_names(self):
            return ["train_zinc.csv", "val_zinc.csv", "test_zinc.csv"]

    @property
    def split_file_name(self):
            return ["train_zinc.csv", "val_zinc.csv", "test_zinc.csv"]

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
            f"{self.split}_real_node_ratio.npy",
        ]

    def download(self):
        from sklearn.model_selection import train_test_split

        def split_smaller_dataset(dataset):
            train_set, temp_set = train_test_split(dataset, test_size=0.2, random_state=42)
            val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)
            return train_set, val_set, test_set

        # Function to split the larger dataset into train and val sizes same as smaller dataset
        def split_larger_dataset(dataset, train_size, val_size):
            train_set, temp_set = train_test_split(dataset, train_size=train_size, random_state=42)
            val_set, test_set = train_test_split(temp_set, train_size=val_size, random_state=42)
            return train_set, val_set, test_set

        import rdkit  # noqa

        data_path = download_url(self.train_url, self.raw_dir)
        os.rename(data_path, osp.join(self.raw_dir, "zinc_250k.csv"))

        data = pd.read_csv(osp.join(self.raw_dir, "zinc_250k.csv"))
        data = data.sample(frac=1, random_state=42)

        if self.transfer:
            source_set = data[(data['logP'] >= 2 - 0.5) & (data['logP'] <= 2 + 0.5)]
            target_set = data[(data['logP'] >= 4 - 0.5) & (data['logP'] <= 4 + 0.5)]

            source_set = source_set[['smiles']]
            target_set = target_set[['smiles']]

            if len(source_set) <= len(target_set):
                train_source, val_source, test_source = split_smaller_dataset(source_set)
                train_target, val_target, test_target = split_larger_dataset(target_set, len(train_source),
                                                                             len(val_source))
            else:
                train_target, val_target, test_target = split_smaller_dataset(target_set)
                train_source, val_source, test_source = split_larger_dataset(source_set, len(train_target),
                                                                             len(val_target))

            if self.is_target:
                files_to_save = [train_target, val_target, test_target]
            else:
                files_to_save = [train_source, val_source, test_source]

            os.remove(osp.join(self.raw_dir, "zinc_250k.csv"))
        else:
            target_set = data[['smiles']]
            train_target, val_target, test_target = split_smaller_dataset(target_set)
            files_to_save = [train_target, val_target, test_target]

        for file, name in zip(files_to_save, self.split_file_name):
            file.columns = ['SMILES']
            file['SMILES'] = file['SMILES'].apply(lambda x: x[0:-1])
            file.to_csv(os.path.join(self.raw_dir, name), index=False)

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


class ZincDataModule(MolecularDataModule):
    def __init__(self, cfg, transfer):
        self.cfg = cfg
        is_target = cfg.is_target
        self.remove_h = False
        if transfer:
            self.datadir = cfg.datadir
            base_path = pathlib.Path(get_original_cwd())
            root_path_source = os.path.join(base_path, self.datadir)
            datasets = {
                "train": ZincDataset(
                    split="train", root=root_path_source, transfer=transfer,
                    is_target=is_target, pre_transform=RemoveYTransform()
                ),
                "val": ZincDataset(
                    split="val", root=root_path_source, transfer=transfer,
                    is_target=is_target, pre_transform=RemoveYTransform()
                ),
                "test": ZincDataset(
                    split="test", root=root_path_source, transfer=transfer,
                    is_target=is_target, pre_transform=RemoveYTransform()
                ),
            }
        else:
            self.datadir = cfg.datadir
            base_path = pathlib.Path(get_original_cwd())
            root_path = os.path.join(base_path, self.datadir)
            datasets = {
                "train": ZincDataset(
                    split="train", root=root_path, transfer=transfer, pre_transform=RemoveYTransform()
                ),
                "val": ZincDataset(
                    split="val", root=root_path, transfer=transfer, pre_transform=RemoveYTransform()
                ),
                "test": ZincDataset(
                    split="test", root=root_path, transfer=transfer, pre_transform=RemoveYTransform()
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


class ZincInfos(AbstractDatasetInfos):
    """
    Moses will not support charge as it only contains one charge type 1
    """

    def __init__(self, datamodule, cfg):
        # basic information
        self.name = "zinc"
        self.is_molecular = True
        self.remove_h = False
        self.use_charge = cfg.use_charge
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
        self.output_dims = PlaceHolder(X=self.num_node_types, charge=0, E=5, y=0)
        if not self.use_charge:
            self.output_dims.charge = 0

        # data specific settings
        self.valencies = [4, 1, 2, 1, 1, 3, 1, 2, 3]
        self.atom_weights = {0: 12, 1: 19, 2: 32, 3: 126.9, 4: 35.4, 5: 31, 6: 79.9, 7: 16, 8: 14}
        self.max_weight = 9 * 80  # Quite arbitrary
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[0:7] = torch.tensor([0.0000, 0.4712, 0.3013, 0.0775, 0.0174, 0.1026, 0.0300])


# if __name__ == "__main__":
#     datadir = '/scratch/uceeosm/DSBConnectivity/data'
#     # base_path = pathlib.Path(get_original_cwd())
#     # root_path = os.path.join(base_path, datadir)
#     zinc = ZincDataset(split='train', root=datadir)
#     zinc.download()


## USE THIS FUNCTION TO GET THE VALENCY DISTRIBUTION GIVEN VALENCIES
# def aggregate_valencies(valencies):
#     max_valency = max(max(val_count.keys()) for val_count in valencies.values())
#     valency_tensor = torch.zeros(max_valency + 1)
#     for atom_type, val_count in valencies.items():
#         for val, count in val_count.items():
#             valency_tensor[int(val)] += count
#     total_valency_count = valency_tensor.sum()
#     if total_valency_count > 0:
#         valency_tensor = valency_tensor / total_valency_count
#
#     return valency_tensor