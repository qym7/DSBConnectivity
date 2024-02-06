
import os
import pathlib
import os.path as osp

import numpy as np
from tqdm import tqdm
import torch
import pickle as pkl
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd
from networkx import to_numpy_array
from torch.utils.data import Dataset

from ..utils import PlaceHolder
from ..datasets.abstract_dataset import (
    AbstractDataModule,
    AbstractDatasetInfos,
)
from ..datasets.dataset_utils import (
    load_pickle,
    save_pickle,
    Statistics,
    to_list,
    RemoveYTransform,
)
from ..metrics.metrics_utils import (
    node_counts,
    atom_type_counts,
    edge_counts,
)
from ..diffusion.distributions import DistributionNodes


class SpectreGraphDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.sbm_file = "sbm_200.pt"
        self.planar_file = "planar_64_200.pt"
        self.comm20_file = "community_12_21_100.pt"
        self.dataset_name = dataset_name

        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.raw_data = torch.load(self.raw_paths[self.file_idx])
        self.max_n_nodes = np.max([d.shape[0] for d in self.raw_data])
        # add one dimension for number of channels
        self.raw_data_to_be_masked = torch.zeros((len(self.raw_data), 1, self.max_n_nodes, self.max_n_nodes))
        for i in range(len(self.raw_data)):
            self.raw_data_to_be_masked[i, 0, :self.raw_data[i].shape[0], :self.raw_data[i].shape[0]] = self.raw_data[i]
        self.num_nodes = self.data.n_nodes
        # import pdb; pdb.set_trace()

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
        )

        # get node dist
        num_nodes = self.statistics.num_nodes
        n_nodes = torch.zeros(max(list(num_nodes.keys())) + 1, dtype=torch.long)
        for key, value in num_nodes.items():
            n_nodes[key] += value
        self.n_nodes = n_nodes / n_nodes.sum()

        self.max_n_nodes = len(n_nodes) - 1
        self.node_dist = DistributionNodes(n_nodes)

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_file_name(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.split == "train":
            return [
                f"train.pt",
                f"train_n.pickle",
                f"train_node_types.npy",
                f"train_bond_types.npy",
            ]
        elif self.split == "val":
            return [
                f"val.pt",
                f"val_n.pickle",
                f"val_node_types.npy",
                f"val_bond_types.npy",
            ]
        else:
            return [
                f"test.pt",
                f"test_n.pickle",
                f"test_node_types.npy",
                f"test_bond_types.npy",
            ]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name == "sbm":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt"
        elif self.dataset_name == "planar":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt"
        elif self.dataset_name == "comm20":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt"
        elif self.dataset_name == "ego":        
            raw_url = "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl"
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        file_path = download_url(raw_url, self.raw_dir)

        if self.dataset_name == 'ego':
            networks = pkl.load(open(file_path, 'rb'))
            adjs = [torch.Tensor(to_numpy_array(network)).fill_diagonal_(0) for network in networks]
        else:
            (
                adjs,
                eigvals,
                eigvecs,
                n_nodes,
                max_eigval,
                min_eigval,
                same_sample,
                n_max,
            ) = torch.load(file_path)
            
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        self.num_graphs = len(adjs)

        if self.dataset_name == 'ego':
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round(self.num_graphs * 0.8))
            val_len = int(round(self.num_graphs * 0.2))
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
            train_indices = indices[:train_len]
            val_indices = indices[:val_len]
            test_indices = indices[train_len:]
        else:
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round((self.num_graphs - test_len) * 0.8))
            val_len = self.num_graphs - train_len - test_len
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
            train_indices = indices[:train_len]
            val_indices = indices[train_len : train_len + val_len]
            test_indices = indices[train_len + val_len :]

        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")
        print(f"Test indices: {test_indices}")
        train_data = []
        val_data = []
        test_data = []
        

        for i, adj in enumerate(adjs):
            # permute randomly nodes as for molecular datasets
            random_order = torch.randperm(adj.shape[-1])
            adj = adj[random_order, :]
            adj = adj[:, random_order]

            if i in train_indices:
                train_data.append(adj)
            if i in val_indices:
                val_data.append(adj)
            if i in test_indices:
                test_data.append(adj)

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        raw_dataset = torch.load(os.path.join(self.raw_dir, "{}.pt".format(self.split)))
        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.long)
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=X.float(), edge_index=edge_index, edge_attr=edge_attr.float(), n_nodes=n_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        num_nodes = node_counts(data_list)
        node_types = atom_type_counts(data_list, num_classes=1)
        bond_types = edge_counts(data_list, num_bond_types=2)
        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_types)
        np.save(self.processed_paths[3], bond_types)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        '''
        return two items to be consistent with other interfaces
        '''
        return self.raw_data_to_be_masked[index], self.num_nodes[index]


# class SpectreGraphDataset(Dataset):
#     def __init__(
#         self,
#         dataset_name,
#         split,
#         root,
#         transform=None,
#         pre_transform=None,
#         pre_filter=None,
#     ):
#         self.sbm_file = "sbm_200.pt"
#         self.planar_file = "planar_64_200.pt"
#         self.comm20_file = "community_12_21_100.pt"
#         self.dataset_name = dataset_name

#         self.split = split
#         if self.split == "train":
#             self.file_idx = 0
#         elif self.split == "val":
#             self.file_idx = 1
#         else:
#             self.file_idx = 2

#         self.raw_data = torch.load(os.path.join(root, "raw/train.pt"))
#         self.max_n_nodes = np.max([d.shape[0] for d in self.raw_data])
#         self.raw_data_to_be_masked = torch.zeros((len(self.raw_data), 1, self.max_n_nodes, self.max_n_nodes))
#         for i in range(len(self.raw_data)):
#             self.raw_data_to_be_masked[i, 0, :self.raw_data[i].shape[0], :self.raw_data[i].shape[0]] = self.raw_data[i]

#         # self.statistics = Statistics(
#         #     num_nodes=load_pickle(self.processed_paths[1]),
#         #     node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
#         #     bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
#         # )

#     @property
#     def raw_file_names(self):
#         return ["train.pt", "val.pt", "test.pt"]

#     def __len__(self):
#         return len(self.raw_data)

#     def __getitem__(self, index):
#         # import pdb; pdb.set_trace()
#         return self.raw_data_to_be_masked[index], torch.tensor(0)
