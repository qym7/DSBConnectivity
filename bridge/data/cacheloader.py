import os
import time

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .. import utils


class CacheLoader(Dataset):
    def __init__(self, fb,
                 sample_net,
                 dataloader_b,
                 num_batches,
                 langevin,
                 n,
                 mean, std,
                 batch_size, device='cpu',
                 dataloader_f=None,
                 transfer=False,
                 graph=False,
                 nodes_dist=None,
                 dataset_infos=None,
                 visualization_tools=None,
                 visualize=False):

        super().__init__()
        self.max_n_nodes = langevin.max_n_nodes
        self.num_steps = langevin.num_steps
        self.num_batches = num_batches
        self.graph = graph
        self.nodes_dist = nodes_dist
        self.device = device
        self.mean = mean
        self.std = std
        self.decart_mean_final = utils.PlaceHolder(
            X=torch.ones(1).to(self.device),
            E=torch.ones(2).to(self.device)*0.5,
            y=None
        )
        self.visualization_tools = visualization_tools
        self.visualize = visualize

        self.data = utils.PlaceHolder(
            X=torch.Tensor(num_batches, batch_size*self.num_steps, 2, self.max_n_nodes, len(dataset_infos.node_types)).to(self.device),
            E=torch.Tensor(num_batches, batch_size*self.num_steps, 2, self.max_n_nodes, self.max_n_nodes, len(dataset_infos.bond_types)).to(self.device),
            y=None
        )
        self.steps_data = torch.zeros(
            (num_batches, batch_size*self.num_steps, 1)).to(device)  # .cpu() # steps
        self.n_nodes = torch.zeros((num_batches, batch_size*self.num_steps))

        with torch.no_grad():
            for b in range(num_batches):
                if (fb == 'b') or (fb == 'f' and transfer):  # actually forward
                    loader = dataloader_b if fb == 'b' else dataloader_f
                    batch = next(loader)
                    batch, node_mask = utils.data_to_dense(batch, self.max_n_nodes)
                    batch = batch.minus(self.decart_mean_final)
                    batch = batch.scale(4)
                    n_nodes = node_mask.sum(-1)
                    batch = batch.mask(node_mask)
                else:
                    n_nodes = self.nodes_dist.sample_n(batch_size, device)
                    batch = utils.PlaceHolder(
                        X=torch.randn(batch_size,
                                       self.max_n_nodes,
                                       len(dataset_infos.node_types)).to(self.device),
                        E=torch.randn(batch_size,
                                       self.max_n_nodes,
                                       self.max_n_nodes,
                                       len(dataset_infos.edge_types)).to(self.device),
                        y=None, charge=None, n_nodes=n_nodes
                    )
                    # batch = batch.scale(std).add(mean)
                    batch.E = utils.symmetize_edge_matrix(batch.E)
                    # arange = (
                    #     torch.arange(self.max_n_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
                    # )   
                    # node_mask = arange < n_nodes.unsqueeze(1)
                    batch.mask()
                    print(b, batch.X[0,0,0])

                if (n == 1) & (fb == 'b'):
                    x, out, steps_expanded = langevin.record_init_langevin(
                        batch, node_mask)
                else:
                    x, out, steps_expanded = langevin.record_langevin_seq(
                        sample_net, batch, node_mask=batch.node_mask, ipf_it=n)

                # if fb == 'b':
                #     if n==2:
                #         import pdb; pdb.set_trace()

                if b == 0 and self.visualize:
                    self.visualize = False
                    print('Visualizing chains...')
                    current_path = os.getcwd()
                    reverse_fb = 'f' if fb == 'b' else 'b'
                    result_path = os.path.join(current_path, f'cache_chains_{reverse_fb}/'
                                                            f'ipf{n}/'
                                                            f'molecule')

                    chain = x.copy()
                    chain.X = torch.concatenate((batch.X.unsqueeze(1), chain.X),dim=1)
                    chain.E = torch.concatenate((batch.E.unsqueeze(1), chain.E),dim=1)
                    _ = self.visualization_tools.visualize_chains(result_path,
                                                                    chains=chain,
                                                                    num_nodes=n_nodes,
                                                                    local_rank=0,
                                                                    num_chains_to_visualize=4,
                                                                    fb=reverse_fb)

                batch_X = torch.cat((x.X.unsqueeze(2), out.X.unsqueeze(2)), dim=2).flatten(start_dim=0, end_dim=1)
                batch_E = torch.cat((x.E.unsqueeze(2), out.E.unsqueeze(2)), dim=2).flatten(start_dim=0, end_dim=1)

                self.data.X[b] = batch_X
                self.data.E[b] = batch_E

                # store steps
                flat_steps = steps_expanded.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps

                # store n_nodes
                n_nodes = n_nodes.unsqueeze(-1).repeat(1, self.num_steps)
                self.n_nodes[b] = n_nodes.flatten()

        self.data = utils.PlaceHolder(
            X=self.data.X.flatten(start_dim=0, end_dim=1),
            E=self.data.E.flatten(start_dim=0, end_dim=1),
            y=None, charge=None
        )

        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)
        self.n_nodes = self.n_nodes.flatten()

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        item = self.data.get_data(index, dim=0)  # X -> (2, max_n_node, dx)
        steps = self.steps_data[index]
        n_nodes = self.n_nodes[index]
        x = item.get_data(0, dim=0)  # X -> (max_n_node, dx)
        out = item.get_data(1, dim=0)

        if x.charge is None:
            x.charge = torch.zeros((x.X.shape[0], 0), device=self.device, dtype=torch.long)
        if out.charge is None:
            out.charge = torch.zeros((out.X.shape[0], 0), device=self.device, dtype=torch.long)
        if x.y is None:
            x.y = torch.zeros((0), device=self.device, dtype=torch.long)
        if out.y is None:
            out.y = torch.zeros((0), device=self.device, dtype=torch.long)

        x = (x.X, x.E, x.y, x.charge, n_nodes)
        out = (out.X, out.E, out.y, out.charge, n_nodes)

        return x, out, steps

    def __len__(self):
        return self.data.X.shape[0]
