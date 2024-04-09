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

from ..utils import PlaceHolder
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
from ..data.synthetic_graphs import (
    generate_sbm_graphs,
    generate_tree_graphs,
    generate_planar_graphs,
)

class SyntheticGraphDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name,
        split,
        root,
        dataset_cfg,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        # track the statistics of the dataset
        root = '/'.join([root, dataset_name])
        #  str(dataset_cfg.min_num_communities,
        #  dataset_cfg.max_num_communities,
        #  dataset_cfg.min_community_size,
        #  dataset_cfg.max_community_size,
        #  dataset_cfg.intra_prob,
        #  dataset_cfg.inter_prob])

        self.split = split
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
        )

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
        if "sbm" in self.dataset_name:
            networks = generate_sbm_graphs(
                num_graphs=self.dataset_cfg.num_graphs,
                min_num_communities=self.dataset_cfg.min_num_communities,
                max_num_communities=self.dataset_cfg.max_num_communities,
                min_community_size=self.dataset_cfg.min_community_size,
                max_community_size=self.dataset_cfg.max_community_size,
                intra_prob=self.dataset_cfg.intra_prob,
                inter_prob=self.dataset_cfg.inter_prob,
                )
            adjs = [torch.Tensor(to_numpy_array(network)).fill_diagonal_(0) for network in networks]
            
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        self.num_graphs = len(adjs)

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


# class SyntheticGraphDataModule(AbstractDataModule):
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.dataset_name = self.cfg.dataset.name
#         self.datadir = cfg.dataset.datadir
#         base_path = pathlib.Path(get_original_cwd()).parents[0]
#         root_path = os.path.join(base_path, self.datadir)
#         pre_transform = RemoveYTransform()

#         datasets = {
#             "train": SyntheticGraphDataset(
#                 dataset_name=self.cfg.dataset.name,
#                 pre_transform=pre_transform,
#                 split="train",
#                 root=root_path,
#             ),
#             "val": SyntheticGraphDataset(
#                 dataset_name=self.cfg.dataset.name,
#                 pre_transform=pre_transform,
#                 split="val",
#                 root=root_path,
#             ),
#             "test": SyntheticGraphDataset(
#                 dataset_name=self.cfg.dataset.name,
#                 pre_transform=pre_transform,
#                 split="test",
#                 root=root_path,
#             ),
#         }

#         self.statistics = {
#             "train": datasets["train"].statistics,
#             "val": datasets["val"].statistics,
#             "test": datasets["test"].statistics,
#         }

#         super().__init__(cfg, datasets)
#         super().prepare_dataloader()
#         self.inter = self.train_dataset


# class SyntheticDatasetInfos(AbstractDatasetInfos):
#     def __init__(self, datamodule):
#         self.is_molecular = False
#         self.spectre = True
#         self.use_charge = False
#         self.dataset_name = datamodule.dataset_name
#         self.node_types = datamodule.inter.statistics.node_types
#         self.bond_types = datamodule.inter.statistics.bond_types
#         super().complete_infos(
#             datamodule.statistics, len(datamodule.inter.statistics.node_types)
#         )
#         self.input_dims = PlaceHolder(
#             X=len(self.node_types), E=len(self.bond_types), y=0, charge=0
#         )
#         self.output_dims = PlaceHolder(
#             X=len(self.node_types), E=len(self.bond_types), y=0, charge=0
#         )
#         self.statistics = {
#             'train': datamodule.statistics['train'],
#             'val': datamodule.statistics['val'],
#             'test': datamodule.statistics['test']
#         }

#     def to_one_hot(self, data):
#         """
#         call in the beginning of data
#         get the one_hot encoding for a charge beginning from -1
#         """
#         data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
#         if data.y is None:
#             data.y = data.x.new_zeros((data.batch.max().item()+1, 0))

#         return data
