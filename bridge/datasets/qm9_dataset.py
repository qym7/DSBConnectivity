import os
import os.path as osp
import pathlib
import sys

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig, QED, MolFromSmiles, MolToSmiles
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from hydra.utils import get_original_cwd

from ..utils import PlaceHolder
from ..datasets.abstract_dataset import (
    MolecularDataModule,
    AbstractDatasetInfos,
)
from ..datasets.dataset_utils import (
    load_pickle,
    save_pickle,
    mol_to_torch_geometric,
    Statistics,
    remove_hydrogens,
    to_list,
    files_exist,
)
from ..metrics.metrics_utils import compute_all_statistics


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


atom_encoder = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
atom_decoder = [key for key in atom_encoder.keys()]


class QM9Dataset(InMemoryDataset):
    raw_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    def __init__(
        self,
        split,
        root,
        transfer,
        is_target,
        remove_h: bool,
        target_prop=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        self.transfer = transfer
        self.is_target = is_target
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        self.target_prop = target_prop

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
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        if self.split == "train":
            return [
                f"train_{h}.pt",
                f"train_n_{h}.pickle",
                f"train_node_types_{h}.npy",
                f"train_bond_types_{h}.npy",
                f"train_charge_{h}.npy",
                f"train_valency_{h}.pickle",
                f"train_smiles_qm9_{h}.pickle",
                f"train_real_node_ratio_{h}.npy",
            ]
        elif self.split == "val":
            return [
                f"val_{h}.pt",
                f"val_n_{h}.pickle",
                f"val_node_types_{h}.npy",
                f"val_bond_types_{h}.npy",
                f"val_charge_{h}.npy",
                f"val_valency_{h}.pickle",
                f"val_smiles_qm9_{h}.pickle",
                f"val_real_node_ratio_{h}.npy",
            ]
        else:
            return [
                f"test_{h}.pt",
                f"test_n_{h}.pickle",
                f"test_node_types_{h}.npy",
                f"test_bond_types_{h}.npy",
                f"test_charge_{h}.npy",
                f"test_valency_{h}.pickle",
                f"test_smiles_qm9_{h}.pickle",
                f"test_real_node_ratio_{h}.npy",
            ]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)
            _ = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                osp.join(self.raw_dir, "3195404"),
                osp.join(self.raw_dir, "uncharacterized.txt"),
            )
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(os.path.join(self.raw_dir, "train_init.csv"))
        val.to_csv(os.path.join(self.raw_dir, "val_init.csv"))
        test.to_csv(os.path.join(self.raw_dir, "test_init.csv"))

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        from sklearn.model_selection import train_test_split

        def split_smaller_dataset(dataset):
            train_set, temp_set = train_test_split(dataset, test_size=0.2, random_state=42)
            val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)
            return train_set, val_set, test_set

        def split_larger_dataset(dataset, train_size, val_size):
            train_set, temp_set = train_test_split(dataset, train_size=train_size, random_state=42)
            val_set, test_set = train_test_split(temp_set, train_size=val_size, random_state=42)
            return train_set, val_set, test_set

        if self.transfer:
            if os.path.exists(self.split_paths[0]):
                print('Files with separated data exists')
            else:
                suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=self.remove_h, sanitize=self.remove_h)
                all_smiles = []
                num_errors = 0
                for i, mol in enumerate(tqdm(suppl)):
                    if mol is None:
                        num_errors += 1
                        continue

                    smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                    if smiles is None:
                        num_errors += 1
                    else:
                        all_smiles.append(smiles)

                print("Number of molecules that could not be mapped to smiles: ", num_errors)
                print('Data separation in progress...')
                sa_greater_3 = []
                sa_less_3 = []

                for smiles in all_smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        sa_score = sascorer.calculateScore(mol)
                        if sa_score <= 3:
                            sa_less_3.append(smiles)
                        elif sa_score > 3:
                            sa_greater_3.append(smiles)

                if len(sa_less_3) <= len(sa_greater_3):
                    train_target, val_target, test_target = split_smaller_dataset(sa_less_3)
                    train_source, val_source, test_source = split_larger_dataset(sa_greater_3, len(train_target),
                                                                                 len(val_target))
                else:
                    train_source, val_source, test_source = split_smaller_dataset(sa_greater_3)
                    train_target, val_target, test_target = split_larger_dataset(sa_less_3, len(train_source),
                                                                                 len(val_source))

                if self.is_target:
                    train_target = pd.DataFrame(train_target, columns=['SMILES'])
                    val_target = pd.DataFrame(val_target, columns=['SMILES'])
                    test_target = pd.DataFrame(test_target, columns=['SMILES'])
                    files_to_save = [train_target, val_target, test_target]
                else:
                    train_source = pd.DataFrame(train_source, columns=['SMILES'])
                    val_source = pd.DataFrame(val_source, columns=['SMILES'])
                    test_source = pd.DataFrame(test_source, columns=['SMILES'])
                    files_to_save = [train_source, val_source, test_source]

                print('Data saving...')
                for file, name in zip(files_to_save, self.split_file_name):
                    file.to_csv(os.path.join(self.raw_dir, name), index=False)

            smile_list = pd.read_csv(self.split_paths[self.file_idx])
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
                data.edge_attr = F.one_hot(data.edge_attr.to(torch.long), num_classes=len(statistics.bond_types)).to(
                    torch.float)
                data.charge = F.one_hot(data.charge.to(torch.long) + 1, num_classes=len(statistics.charge_types[0])).to(
                    torch.float)
            torch.save(self.collate(data_list), self.processed_paths[0])
            np.save(self.processed_paths[7], statistics.real_node_ratio)
        else:
            target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
            target_df.drop(columns=["mol_id"], inplace=True)

            with open(self.raw_paths[-1], "r") as f:
                skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

            suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=self.remove_h, sanitize=self.remove_h)
            data_list = []
            all_smiles = []
            num_errors = 0
            for i, mol in enumerate(tqdm(suppl)):
                if i in skip or i not in target_df.index:
                    continue

                if mol is None:
                    num_errors += 1
                    continue

                smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                if smiles is None:
                    num_errors += 1
                else:
                    all_smiles.append(smiles)

                data = mol_to_torch_geometric(mol, atom_encoder, smiles)
                if self.remove_h:
                    data = remove_hydrogens(data)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # data.edge_attr = F.one_hot(data.edge_attr, num_classes=5).to(torch.float)
                # data.x = F.one_hot(data.x, num_classes=len(list(self.atom_encoder.keys()))).to(torch.float)

                if data.edge_index.numel() > 0:
                    data_list.append(data)

            statistics = compute_all_statistics(
                data_list, self.atom_encoder, charge_dic={-1: 0, 0: 1, 1: 2}
            )
            save_pickle(statistics.num_nodes, self.processed_paths[1])
            np.save(self.processed_paths[2], statistics.node_types)
            np.save(self.processed_paths[3], statistics.bond_types)
            np.save(self.processed_paths[4], statistics.charge_types)
            save_pickle(statistics.valencies, self.processed_paths[5])
            save_pickle(set(all_smiles), self.processed_paths[6])
            np.save(self.processed_paths[7], statistics.real_node_ratio)

            for data in data_list:
                data.x = F.one_hot(data.x.to(torch.long), num_classes=len(statistics.node_types)).to(torch.float)
                data.edge_attr = F.one_hot(data.edge_attr.to(torch.long), num_classes=len(statistics.bond_types)).to(
                    torch.float)
                data.charge = F.one_hot(data.charge.to(torch.long) + 1, num_classes=len(statistics.charge_types[0])).to(
                    torch.float)

            torch.save(self.collate(data_list), self.processed_paths[0])
            print("Number of molecules that could not be mapped to smiles: ", num_errors)


class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg, transfer):
        self.cfg = cfg
        self.datadir = cfg.datadir
        is_target = cfg.is_target
        base_path = pathlib.Path(get_original_cwd())
        root_path = os.path.join(base_path, self.datadir)

        # target = getattr(cfg.general, "guidance_target", None)
        # regressor = getattr(self, "regressor", None)
        target = regressor = None
        if regressor and target == "mu":
            transform = SelectMuTransform()
        elif regressor and target == "homo":
            transform = SelectHOMOTransform()
        elif regressor and target == "both":
            transform = None
        else:
            transform = RemoveYTransform()

        self.remove_h = cfg.remove_h
        datasets = {
            "train": QM9Dataset(
                split="train",
                root=root_path,
                transfer=transfer,
                is_target=is_target,
                remove_h=self.cfg.remove_h,
                target_prop=target,
                transform=RemoveYTransform(),
            ),
            "val": QM9Dataset(
                split="val",
                root=root_path,
                transfer=transfer,
                is_target=is_target,
                remove_h=self.cfg.remove_h,
                target_prop=target,
                transform=RemoveYTransform(),
            ),
            "test": QM9Dataset(
                split="test",
                root=root_path,
                transfer=transfer,
                is_target=is_target,
                remove_h=self.cfg.remove_h,
                target_prop=target,
                transform=transform,
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


class QM9Infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        # basic settings
        self.name = "qm9"
        self.is_molecular = True
        self.remove_h = cfg.remove_h
        self.use_charge = cfg.use_charge
        self.collapse_charges = torch.Tensor([-1, 0, 1]).int()

        # statistics
        self.statistics = datamodule.statistics
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
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
        # dimensions settings
        self.output_dims = PlaceHolder(
            X=self.num_node_types, charge=self.num_charge_types, E=5, y=0
        )
        if not self.use_charge:
            self.output_dims.charge = 0

        # special settings
        # atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.valencies = [4, 3, 2, 1] if self.remove_h else [1, 4, 3, 2, 1]
        # self.atom_weights = [12, 14, 16, 19] if self.remove_h else [1, 12, 14, 16, 19]
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