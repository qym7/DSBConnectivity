import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time

from ..utils import get_masks


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
                 node_dist=None):

        super().__init__()
        start = time.time()
        shape = langevin.d
        num_steps = langevin.num_steps
        self.graph = graph
        self.node_dist = node_dist
        self.data = torch.zeros(
            (num_batches, batch_size*num_steps, 2, *shape)).to(device)  # .cpu()
        # self.steps_data = torch.zeros(
        #     (num_batches, batch_size*num_steps, 1), dtype=torch.long).to(device)  # .cpu() # steps
        self.steps_data = torch.zeros(
            (num_batches, batch_size*num_steps, 1)).to(device)  # .cpu() # steps
        
        if self.graph:
            self.n_nodes = torch.zeros((num_batches, batch_size, num_steps))

        with torch.no_grad():
            for b in range(num_batches):
                if fb == 'b':  # actually forward
                    batch = next(dataloader_b)
                    if self.graph:
                        batch, n_nodes = batch
                        batch = (batch.to(device), n_nodes.to(device))
                    else:
                        batch = batch[0].to(device)
                elif fb == 'f' and transfer:  # actually backward
                    batch = next(dataloader_f)
                    if self.graph:
                        batch, n_nodes = batch
                        batch = (batch.to(device), n_nodes.to(device))
                    else:
                        batch = batch[0].to(device)
                else:
                    # batch = next(dataloader_f)
                    # use dataloader to create number of nodes
                    if self.graph:
                        n_nodes = self.node_dist.sample_n(batch_size, device)
                        _, edge_mask = get_masks(
                            n_nodes, self.node_dist.max_n_nodes, batch_size, device)
                    batch = mean + std * \
                        torch.randn((batch_size, *shape), device=device)
                    if self.graph:
                        batch = batch * edge_mask.unsqueeze(1)
                        batch = (batch, n_nodes.to(device))

                if (n == 1) & (fb == 'b'):
                    x, out, steps_expanded = langevin.record_init_langevin(
                        batch)
                else:
                    x, out, steps_expanded = langevin.record_langevin_seq(
                        sample_net, batch, ipf_it=n)

                if self.graph:
                    self.n_nodes[b] = x[1].unsqueeze(-1).repeat(1, num_steps)
                    x = x[0]  # mask out n_nodes which is wrapped with x

                # store x, out
                x = x.unsqueeze(2)
                out = out.unsqueeze(2)
                batch_data = torch.cat((x, out), dim=2)
                flat_data = batch_data.flatten(start_dim=0, end_dim=1)
                self.data[b] = flat_data

                # store steps
                flat_steps = steps_expanded.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps

        if self.graph:
            self.n_nodes = self.n_nodes.flatten()
        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

        stop = time.time()
        print('Cache size: {0}'.format(self.data.shape))
        print("Load time: {0}".format(stop-start))

    def __getitem__(self, index):
        item = self.data[index]
        x = item[0]
        out = item[1]
        steps = self.steps_data[index]
        
        if self.graph:
            x = (x, self.n_nodes[index])

        return x, out, steps

    def __len__(self):
        return self.data.shape[0]
