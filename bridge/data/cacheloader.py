import os
import time

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb

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
                 visualize=False,
                 scale=1,
                 decart_mean_final=None):

        super().__init__()
        self.max_n_nodes = langevin.max_n_nodes
        self.num_steps = langevin.num_steps
        self.num_batches = num_batches
        self.graph = graph
        self.nodes_dist = nodes_dist
        self.device = device
        self.mean = mean
        self.std = std
        self.scale = scale
        self.decart_mean_final = decart_mean_final
        self.visualization_tools = visualization_tools
        self.visualize = visualize

        self.data = utils.PlaceHolder(
            X=torch.Tensor(num_batches, batch_size*self.num_steps, 2, self.max_n_nodes, len(dataset_infos.node_types)).to(self.device),
            E=torch.Tensor(num_batches, batch_size*self.num_steps, 2, self.max_n_nodes, self.max_n_nodes, len(dataset_infos.edge_types)).to(self.device),
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
                    batch = batch.scale(self.scale)
                    n_nodes = node_mask.sum(-1)
                    # batch.X = torch.zeros_like(batch.X, device=batch.X.device)
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
                    batch = batch.scale(self.std).add(self.mean)
                    batch.E = utils.symmetize_edge_matrix(batch.E)
                    batch.mask()

                if (n == 1) & (fb == 'b'):
                    x, out, steps_expanded = langevin.record_init_langevin(
                        batch, node_mask)
                else:
                    x, out, steps_expanded = langevin.record_langevin_seq(
                        sample_net, batch, node_mask=batch.node_mask, ipf_it=n, fb=fb)

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
                                                                    num_chains_to_visualize=1,
                                                                    fb='f')

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

        fb_r = 'f_learn_b' if fb == 'b' else 'b_learn_f'
        shape_E = self.data.E.shape
        E_f = self.data.E.reshape(shape_E[0], batch_size, self.num_steps, shape_E[-4], shape_E[-3], shape_E[-2], shape_E[-1])[0,:,:,0,:,:]
        E_f = E_f[..., 0].mean(-1).mean(-1).mean(0).cpu().numpy() - E_f[..., 1].mean(-1).mean(-1).mean(0).cpu().numpy()
        E_f = E_f[::-1]
        dataa = [[x, y] for (x, y) in zip(torch.arange(E_f.shape[0]), E_f)]
        table = wandb.Table(data=dataa, columns=["steps", "mean"])
        wandb.log({f'{fb}_{n}_mean_true': wandb.plot.line(table, "steps", "mean", title=f'{fb}_{n}_mean_true')})

        self.data = utils.PlaceHolder(
            X=self.data.X.flatten(start_dim=0, end_dim=1),
            E=self.data.E.flatten(start_dim=0, end_dim=1),
            y=None, charge=None
        )

        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)
        self.n_nodes = self.n_nodes.flatten()

    def __getitem__(self, index):
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


def DBDSB_CacheLoader(
                    sample_direction,
                    sample_fn,
                    init_dl,
                    final_dl,
                    num_batches,
                    langevin,
                    ipf,
                    n,
                    refresh_idx=0,
                    refresh_tot=1,
                    device='cpu'
    ):
    start = time.time()

    # New method, saving as npy
    cache_filename_npy = f'cache_{sample_direction}_{n:03}.npy'
    cache_filepath_npy = os.path.join(ipf.cache_dir, cache_filename_npy)

    cache_filename_txt = f'cache_{sample_direction}_{n:03}.txt'
    cache_filepath_txt = os.path.join(ipf.cache_dir, cache_filename_txt)

    if ipf.cdsb:
        cache_y_filename_npy = f'cache_y_{sample_direction}_{n:03}.npy'
        cache_y_filepath_npy = os.path.join(ipf.cache_dir, cache_y_filename_npy)

    # Temporary cache of each batch
    temp_cache_dir = os.path.join(ipf.cache_dir, f"temp_{sample_direction}_{n:03}_{refresh_idx:03}")
    os.makedirs(temp_cache_dir, exist_ok=True)

    npar = num_batches * ipf.cache_batch_size
    num_batches_dist = num_batches * ipf.accelerator.num_processes  # In distributed mode
    cache_batch_size_dist = ipf.cache_batch_size // ipf.accelerator.num_processes  # In distributed mode

    use_existing_cache = False
    if os.path.isfile(cache_filepath_txt):
        f = open(cache_filepath_txt, 'r')
        input = f.readline()
        f.close()
        input_list = input.split("/")
        if int(input_list[0]) == refresh_idx and int(input_list[1]) == refresh_tot:
            use_existing_cache = True
    
    if not use_existing_cache:
        sample = ((sample_direction == 'b') or ipf.transfer)
        normalize_x1 = ((not sample) and ipf.normalize_x1)

        x1_mean_list, x1_mse_list = [], []

        for b in range(num_batches):
            b_dist = b * ipf.accelerator.num_processes + ipf.accelerator.process_index

            try:
                batch_x0, batch_x1 = torch.load(os.path.join(temp_cache_dir, f"{b_dist}.pt"))
                assert len(batch_x0) == len(batch_x1) == cache_batch_size_dist
                batch_x0, batch_x1 = batch_x0.to(ipf.device), batch_x1.to(ipf.device)
            except:
                ipf.set_seed(seed=ipf.compute_current_step(0, n+1)*num_batches_dist*refresh_tot + num_batches_dist*refresh_idx + b_dist)

                init_batch_x, init_batch_y, final_batch_x, _, _ = ipf.sample_batch(init_dl, final_dl)

                with torch.no_grad():
                    batch_x0, batch_y, batch_x1 = langevin.generate_new_dataset(init_batch_x, init_batch_y, final_batch_x, sample_fn, sample_direction, sample=sample, num_steps=ipf.cache_num_steps)
                    batch_x0, batch_x1 = batch_x0.contiguous(), batch_x1.contiguous()
                    torch.save([batch_x0, batch_x1], os.path.join(temp_cache_dir, f"{b_dist}.pt"))
                    if ipf.cdsb:
                        torch.save([batch_y], os.path.join(temp_cache_dir, f"{b_dist}_y.pt"))

            if normalize_x1:
                x1_mean_list.append(batch_x1.mean(0))
                x1_mse_list.append(batch_x1.square().mean(0))
        
        if normalize_x1:
            x1_mean = torch.stack(x1_mean_list).mean(0)
            x1_mse = torch.stack(x1_mse_list).mean(0)
            reduced_x1_mean = ipf.accelerator.reduce(x1_mean, reduction='mean')
            reduced_x1_mse = ipf.accelerator.reduce(x1_mse, reduction='mean')
            reduced_x1_std = (reduced_x1_mse - reduced_x1_mean.square()).sqrt()
    
        ipf.accelerator.wait_for_everyone() 

        stop = time.time()
        ipf.accelerator.print("Load time: {0}".format(stop-start))

        # Aggregate temporary caches into central cache file
        if ipf.accelerator.is_main_process:
            fp = open_memmap(cache_filepath_npy, dtype='float32', mode='w+', shape=(npar, 2, *batch_x0.shape[1:]))
            if ipf.cdsb:
                fp_y = open_memmap(cache_y_filepath_npy, dtype='float32', mode='w+', shape=(npar, 1, *batch_y.shape[1:]))
            for b_dist in range(num_batches_dist):
                temp_cache_filepath_b_dist = os.path.join(temp_cache_dir, f"{b_dist}.pt")
                loaded = False
                while not loaded:
                    if not os.path.isfile(temp_cache_filepath_b_dist):
                        print(f"Index {ipf.accelerator.process_index} did not find temp cache file {b_dist}, retrying in 5 seconds")
                        time.sleep(5)
                    else:
                        try:
                            batch_x0, batch_x1 = torch.load(temp_cache_filepath_b_dist)
                            batch_x0, batch_x1 = batch_x0.to(ipf.device), batch_x1.to(ipf.device)
                            loaded = True
                        except:
                            print(f"Index {ipf.accelerator.process_index} failed to load cache file {b_dist}, retrying in 5 seconds")
                            time.sleep(5)

                assert len(batch_x0) == len(batch_x1) == cache_batch_size_dist

                if ipf.cdsb:
                    temp_cache_y_filepath_b_dist = os.path.join(temp_cache_dir, f"{b_dist}_y.pt")
                    loaded = False
                    while not loaded:
                        if not os.path.isfile(temp_cache_y_filepath_b_dist):
                            print(f"Index {ipf.accelerator.process_index} did not find temp cache file {b_dist}_y, retrying in 5 seconds")
                            time.sleep(5)
                        else:
                            try:
                                batch_y = torch.load(temp_cache_y_filepath_b_dist)[0]
                                loaded = True
                            except:
                                print(f"Index {ipf.accelerator.process_index} failed to load cache file {b_dist}_y, retrying in 5 seconds")
                                time.sleep(5)
                    assert len(batch_y) == cache_batch_size_dist
                
                if normalize_x1:
                    batch_x1 = (batch_x1 - reduced_x1_mean) / reduced_x1_std
    
                batch = torch.stack([batch_x0, batch_x1], dim=1).float().cpu().numpy()
                fp[b_dist*cache_batch_size_dist:(b_dist+1)*cache_batch_size_dist] = batch
                fp.flush()

                if ipf.cdsb:
                    batch_y = batch_y.unsqueeze(1).float().cpu().numpy()
                    fp_y[b_dist*cache_batch_size_dist:(b_dist+1)*cache_batch_size_dist] = batch_y
                    fp_y.flush()
            
            del fp
            if ipf.cdsb:
                del fp_y
                
            f = open(cache_filepath_txt, 'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
            f.write(f'{refresh_idx}/{refresh_tot}')
            f.close()

            shutil.rmtree(temp_cache_dir)
            
    ipf.accelerator.wait_for_everyone() 

    # All processes check that the cache is accessible
    loaded = False
    while not loaded:
        if not os.path.isfile(cache_filepath_npy):
            print("Index", ipf.accelerator.process_index, "did not find cache file, retrying in 5 seconds")
            time.sleep(5)
        else:
            try:
                fp = np.load(cache_filepath_npy, mmap_mode='r')
                loaded = True
            except:
                print("Index", ipf.accelerator.process_index, "failed to load cache file, retrying in 5 seconds")
                time.sleep(5)
    
    if ipf.cdsb:
        loaded = False
        while not loaded:
            if not os.path.isfile(cache_y_filepath_npy):
                print("Index", ipf.accelerator.process_index, "did not find cache_y file, retrying in 5 seconds")
                time.sleep(5)
            else:
                try:
                    fp_y = np.load(cache_y_filepath_npy, mmap_mode='r')
                    loaded = True
                except:
                    print("Index", ipf.accelerator.process_index, "failed to load cache_y file, retrying in 5 seconds")
                    time.sleep(5)

    ipf.accelerator.wait_for_everyone() 
    ipf.accelerator.print(f'Cache size: {fp.shape}')

    if ipf.accelerator.is_main_process:
        # Visualize first entries
        num_plots_grid = 100
        ipf.plotter.save_image(torch.from_numpy(fp[:num_plots_grid, 0]), f'cache_{sample_direction}_{n:03}_x0', "./", domain=0)
        ipf.plotter.save_image(torch.from_numpy(fp[:num_plots_grid, 1]), f'cache_{sample_direction}_{n:03}_x1', "./", domain=1)

        # Automatically delete old cache files
        for fb in ['f', 'b']:
            existing_cache_files = sorted(glob.glob(os.path.join(ipf.cache_dir, f"cache_{fb}_**.npy")))
            for ckpt_i in range(max(len(existing_cache_files)-1, 0)):
                if not os.path.samefile(existing_cache_files[ckpt_i], cache_filepath_npy):
                    os.remove(existing_cache_files[ckpt_i])

            if ipf.cdsb:
                existing_cache_files = sorted(glob.glob(os.path.join(ipf.cache_dir, f"cache_y_{fb}_**.npy")))
                for ckpt_i in range(max(len(existing_cache_files)-1, 0)):
                    if not os.path.samefile(existing_cache_files[ckpt_i], cache_filepath_npy):
                        os.remove(existing_cache_files[ckpt_i])

    del fp

    if ipf.cdsb:
        del fp_y
        return MemMapTensorDataset([cache_filepath_npy, cache_y_filepath_npy])

    return MemMapTensorDataset([cache_filepath_npy])