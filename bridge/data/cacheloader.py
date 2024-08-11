import os
import time

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .. import utils


class CacheLoader(Dataset):
    def __init__(
        self,
        fb,
        sample_net,
        dataloader_b,
        num_batches,
        langevin,
        n,
        batch_size,
        device="cpu",
        limit_dist=None,
        dataloader_f=None,
        transfer=False,
        graph=False,
        nodes_dist=None,
        dataset_infos=None,
        visualization_tools=None,
        visualize=False,
        virtual_node=False,
    ):
        super().__init__()
        self.max_n_nodes = langevin.max_n_nodes
        self.num_steps = langevin.num_steps
        self.num_batches = num_batches
        self.graph = graph
        self.nodes_dist = nodes_dist
        self.device = device
        self.visualization_tools = visualization_tools
        self.visualize = visualize
        self.virtual_node = virtual_node

        self.limit_dist = limit_dist

        self.data = utils.PlaceHolder(
            X=torch.Tensor(
                num_batches,
                batch_size * self.num_steps,
                3,
                self.max_n_nodes,
                len(dataset_infos.node_types) + 1 if virtual_node else len(dataset_infos.node_types),
            ).to(self.device),
            E=torch.Tensor(
                num_batches,
                batch_size * self.num_steps,
                3,
                self.max_n_nodes,
                self.max_n_nodes,
                len(dataset_infos.edge_types),
            ).to(self.device),
            y=None,
        )
        self.times_data = torch.zeros((num_batches, batch_size * self.num_steps, 1)).to(
            device
        )  # .cpu() # steps
        self.gammas_data = torch.zeros(
            (num_batches, batch_size * self.num_steps, 1)
        ).to(
            device
        )  # .cpu() # steps
        self.n_nodes = torch.zeros((num_batches, batch_size * self.num_steps))

        with torch.no_grad():
            for b in range(num_batches):
                if (fb == "b") or (fb == "f" and transfer):  # actually forward
                    loader = dataloader_b if fb == "b" else dataloader_f
                    batch = next(loader)
                    batch, node_mask = utils.data_to_dense(batch, self.max_n_nodes)
                    n_nodes = node_mask.sum(-1)
                    if self.virtual_node:
                        batch = utils.add_virtual_node(batch, node_mask)
                        node_mask = torch.ones_like(node_mask).to(batch.X.device).bool()

                else:
                    n_nodes = self.nodes_dist.sample_n(batch_size, device)
                    arange = (
                        torch.arange(self.max_n_nodes, device=self.device)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )
                    node_mask = arange < n_nodes.unsqueeze(1)
                    if self.virtual_node:
                        node_mask = torch.ones_like(node_mask).to(self.device).bool()
                        n_nodes = node_mask.sum(-1)

                    batch = utils.PlaceHolder(
                        X=self.limit_dist.X.repeat(batch_size, self.max_n_nodes, 1).to(
                            self.device
                        ),
                        E=self.limit_dist.E.repeat(
                            batch_size, self.max_n_nodes, self.max_n_nodes, 1
                        ).to(self.device),
                        y=None,
                        charge=None,
                        n_nodes=n_nodes,
                    )

                    batch = batch.sample(onehot=True, node_mask=node_mask)

                batch.mask(node_mask)

                if (n == 1) & (fb == "b"):
                    (
                        x,
                        out,
                        gammas_expanded,
                        times_expanded,
                    ) = langevin.record_init_langevin(batch, node_mask)
                else:
                    (
                        x,
                        out,
                        gammas_expanded,
                        times_expanded,
                    ) = langevin.record_langevin_seq(
                        sample_net, batch, node_mask=batch.node_mask, ipf_it=n
                    )

                batch_X = torch.cat(
                    (x.X.unsqueeze(2), out.X.unsqueeze(2), batch.X.unsqueeze(1).repeat(1, x.X.shape[1], 1, 1).unsqueeze(2)), dim=2
                ).flatten(start_dim=0, end_dim=1)
                batch_E = torch.cat(
                    (x.E.unsqueeze(2), out.E.unsqueeze(2), batch.E.unsqueeze(1).repeat(1, x.E.shape[1], 1, 1, 1).unsqueeze(2)), dim=2
                ).flatten(start_dim=0, end_dim=1)

                try:
                    self.data.X[b] = batch_X
                    self.data.E[b] = batch_E
                except:
                    import pdb; pdb.set_trace()

                # store steps
                flat_times = times_expanded.flatten(start_dim=0, end_dim=1)
                self.times_data[b] = flat_times

                # store gammas
                flat_gammas = gammas_expanded.flatten(start_dim=0, end_dim=1)
                self.gammas_data[b] = flat_gammas

                # store n_nodes
                n_nodes = n_nodes.unsqueeze(-1).repeat(1, self.num_steps)
                self.n_nodes[b] = n_nodes.flatten()

        self.data = utils.PlaceHolder(
            X=self.data.X.flatten(start_dim=0, end_dim=1),
            E=self.data.E.flatten(start_dim=0, end_dim=1),
            y=None,
            charge=None,
        )

        self.times_data = self.times_data.flatten(start_dim=0, end_dim=1)
        self.gammas_data = self.gammas_data.flatten(start_dim=0, end_dim=1)
        self.n_nodes = self.n_nodes.flatten()

    def __getitem__(self, index):
        item = self.data.get_data(index, dim=0)  # X -> (2, max_n_node, dx)
        times = self.times_data[index]
        gammas = self.gammas_data[index]
        n_nodes = self.n_nodes[index]
        x = item.get_data(0, dim=0)  # X -> (max_n_node, dx)
        out = item.get_data(1, dim=0)
        clean = item.get_data(2, dim=0)

        if x.charge is None:
            x.charge = torch.zeros(
                (x.X.shape[0], 0), device=self.device, dtype=torch.long
            )
        if out.charge is None:
            out.charge = torch.zeros(
                (out.X.shape[0], 0), device=self.device, dtype=torch.long
            )
        if x.y is None:
            x.y = torch.zeros((0), device=self.device, dtype=torch.long)
        if out.y is None:
            out.y = torch.zeros((0), device=self.device, dtype=torch.long)
        if clean.y is None:
            clean.y = torch.zeros((0), device=self.device, dtype=torch.long)
        if clean.y is None:
            clean.y = torch.zeros((0), device=self.device, dtype=torch.long)

        clean = (clean.X, clean.E, x.y, x.charge, n_nodes)
        x = (x.X, x.E, x.y, x.charge, n_nodes)
        out = (out.X, out.E, out.y, out.charge, n_nodes)

        return x, out, clean, gammas, times

    def __len__(self):
        return self.data.X.shape[0]
