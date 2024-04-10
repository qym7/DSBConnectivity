
import os
import sys
import time
import json
import random
import datetime

import wandb
import torch
import torch.nn.functional as F
import numpy as np
# from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.autograd.profiler as profiler
from torch.utils.data import WeightedRandomSampler
from accelerate import Accelerator, DistributedType
# from torch_geometric.loader import DataLoader as pygloader.DataLoader
import torch_geometric.loader as pygloader

from .ema import EMAHelper
from . import repeater
from .config_getters import * # get_models, get_graph_models, get_optimizers, get_datasets, get_plotter, get_logger
from .. import utils
from ..data import CacheLoader
from ..langevin import Langevin
from ..metrics.sampling_metrics import SamplingMetrics
from ..analysis.visualization import Visualizer

from ..diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from ..diffusion.extra_features_molecular import ExtraMolecularFeatures



def setup_wandb(cfg):
    kwargs = {
        "name": cfg.project_name,
        "project": f"DSB_{cfg.Dataset}",
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.wandb,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")


class IPFBase(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        setup_wandb(args)
        self.args = args

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(fp16=False, cpu=args.device == 'cpu')
        self.device = self.accelerator.device  # torch.device(args.device)

        # training params
        self.n_ipf = self.args.n_ipf
        self.num_steps = self.args.num_steps
        self.batch_size = self.args.batch_size
        self.num_iter = self.args.num_iter
        self.grad_clipping = self.args.grad_clipping
        self.fast_sampling = self.args.fast_sampling
        self.lr = self.args.lr
        self.graph = self.args.graph

        n = self.num_steps//2
        if self.args.gamma_space == 'linspace':
            gamma_half = np.linspace(self.args.gamma_min, args.gamma_max, n)
        elif self.args.gamma_space == 'geomspace':
            gamma_half = np.geomspace(
                self.args.gamma_min, self.args.gamma_max, n)
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        gammas = torch.tensor(gammas).to(self.device)
        self.T = torch.sum(gammas)  # T is more dynamic
        # import pdb; pdb.set_trace()
        self.current_epoch = 0  # TODO: this need to be changed learning

        # get loggers
        self.logger = self.get_logger()
        self.save_logger = self.get_logger('plot_logs')

        # get data
        print('creating dataloaders')
        # dataloaders = self.build_dataloaders()
        dataloaders = self.build_datamodules()
        self.max_n_nodes = self.datainfos.max_n_nodes
        self.visualization_tools = Visualizer(dataset_infos=self.datainfos, thres=args.thres)

        # create metrics for graph dataset
        if self.graph:
            print('creating metrics for graph dataset')
            # create the training/test/val dataloader
            self.val_sampling_metrics = SamplingMetrics(
                dataset_infos=self.datainfos, test=False, dataloaders=dataloaders
            )
            self.test_sampling_metrics = SamplingMetrics(
                dataset_infos=self.datainfos, test=True, dataloaders=dataloaders
            )

        # get models
        print('building models')
        self.build_models()
        self.build_ema()

        # get optims
        self.build_optimizers()

        # langevin
        if self.args.weight_distrib:
            alpha = self.args.weight_distrib_alpha
            prob_vec = (1 + alpha) * torch.sum(gammas) - \
                torch.cumsum(gammas, 0)
        else:
            prob_vec = gammas * 0 + 1
        time_sampler = torch.distributions.categorical.Categorical(prob_vec)

        max_n_nodes = self.datainfos.max_n_nodes    
        print('creating Langevin')
        self.langevin = Langevin(self.num_steps, max_n_nodes, gammas,
                                 time_sampler, device=self.device,
                                 mean_final=self.mean_final, var_final=self.var_final,
                                 mean_match=self.args.mean_match, graph=self.graph,
                                 extra_features=self.extra_features,
                                 domain_features=self.domain_features,
                                 tf_extra_features=self.tf_extra_features,
                                 tf_domain_features=self.tf_domain_features)

        # checkpoint
        date = str(datetime.datetime.now())[0:10]
        self.name_all = date + self.args.project_name

        # run from checkpoint
        self.checkpoint_run = self.args.checkpoint_run
        if self.args.checkpoint_run:
            self.checkpoint_it = self.args.checkpoint_it
            self.checkpoint_pass = self.args.checkpoint_pass
        else:
            self.checkpoint_it = 1
            self.checkpoint_pass = 'b'

        self.plotter = self.get_plotter()

        if self.accelerator.process_index == 0:
            if not os.path.exists('./im'):
                os.mkdir('./im')
            # if not os.path.exists('./gif'):
            #     os.mkdir('./gif')
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')

        self.stride = self.args.gif_stride
        self.stride_log = self.args.log_stride

    def get_logger(self, name='logs'):
        return get_logger(self.args, name)

    def get_plotter(self):
        return get_plotter(self, self.args)

    def build_models(self, forward_or_backward=None):
        # running network
        net_f, net_b = get_graph_models(self.args, self.datainfos)

        if self.args.checkpoint_run:
            if "checkpoint_f" in self.args:
                net_f.load_state_dict(torch.load(self.args.checkpoint_f))
            if "checkpoint_b" in self.args:
                net_b.load_state_dict(torch.load(self.args.checkpoint_b))

        if self.args.dataparallel:
            net_f = torch.nn.DataParallel(net_f)
            net_b = torch.nn.DataParallel(net_b)

        if forward_or_backward is None:
            net_f = net_f.to(self.device)
            net_b = net_b.to(self.device)
            self.net = torch.nn.ModuleDict({'f': net_f, 'b': net_b})
        if forward_or_backward == 'f':
            net_f = net_f.to(self.device)
            self.net.update({'f': net_f})
        if forward_or_backward == 'b':
            net_b = net_b.to(self.device)
            self.net.update({'b': net_b})

    def accelerate(self, forward_or_backward):
        (self.net[forward_or_backward], self.optimizer[forward_or_backward]) = self.accelerator.prepare(
            self.net[forward_or_backward], self.optimizer[forward_or_backward])

    def update_ema(self, forward_or_backward):
        if self.args.ema:
            self.ema_helpers[forward_or_backward] = EMAHelper(
                mu=self.args.ema_rate, device=self.device)
            self.ema_helpers[forward_or_backward].register(
                self.net[forward_or_backward])

    def build_ema(self):
        if self.args.ema:
            self.ema_helpers = {}
            self.update_ema('f')
            self.update_ema('b')

            if self.args.checkpoint_run:
                # sample network
                sample_net_f, sample_net_b = get_graph_models(self.args, self.datainfos)

                if "sample_checkpoint_f" in self.args:
                    sample_net_f.load_state_dict(
                        torch.load(self.args.sample_checkpoint_f))
                    if self.args.dataparallel:
                        sample_net_f = torch.nn.DataParallel(sample_net_f)
                    sample_net_f = sample_net_f.to(self.device)
                    self.ema_helpers['f'].register(sample_net_f)
                if "sample_checkpoint_b" in self.args:
                    sample_net_b.load_state_dict(
                        torch.load(self.args.sample_checkpoint_b))
                    if self.args.dataparallel:
                        sample_net_b = torch.nn.DataParallel(sample_net_b)
                    sample_net_b = sample_net_b.to(self.device)
                    self.ema_helpers['b'].register(sample_net_b)

    def build_optimizers(self):
        optimizer_f, optimizer_b = get_optimizers(
            self.net['f'], self.net['b'], self.lr)
        optimizer_b = optimizer_b
        optimizer_f = optimizer_f
        self.optimizer = {'f': optimizer_f, 'b': optimizer_b}

    def get_final_stats(self, init_ds):
        '''
        marginal of the distribution should be the starting point
        this function is not correct due to the usage of graph datasets
        '''
        if self.args.adaptive_mean:
            NAPPROX = 100
            # TODO: this batch size causes error when it can not be devided by the datasize
            import pdb; pdb.set_trace()
            vec = next(iter(pygloader.DataLoader(init_ds, batch_size=NAPPROX)))[0]
            mean_final = vec.mean()
            mean_final = vec[0] * 0 + mean_final
            var_final = eval(self.args.var_final)
            final_ds = None
        elif self.args.final_adaptive:
            NAPPROX = 100
            vec = next(iter(pygloader.DataLoader(init_ds, batch_size=NAPPROX)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var()
            final_ds = None
        else:
            mean_final = eval(self.args.mean_final)
            var_final = eval(self.args.var_final)
            final_ds = None
        
        mean_final = mean_final.to(self.device)
        val_final = var_final.to(self.device)
        std_final = torch.sqrt(var_final).to(self.device)
        
        return mean_final, val_final, std_final

    def get_extra_features(self, dataset_infos, datamodule, domain_features):
        ef = self.args.model.extra_features
        edge_f = self.args.model.edge_features
        extra_features = (
            ExtraFeatures(
                eigenfeatures=self.args.model.eigenfeatures,
                edge_features_type=edge_f,
                dataset_info=dataset_infos,
                num_eigenvectors=self.args.model.num_eigenvectors,
                num_eigenvalues=self.args.model.num_eigenvalues,
                num_degree=self.args.model.num_degree,
                dist_feat=self.args.model.dist_feat,
                use_positional=self.args.model.positional_encoding
            )
            if ef is not None
            else DummyExtraFeatures()
        )

        dataset_infos.compute_input_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        return extra_features, dataset_infos

    def build_datamodules(self):
        # get datamodules for train/val/test data
        out = get_both_datamodules(self.args)
        self.train_metrics, self.domain_features, self.datamodule, self.datainfos, \
            self.tf_train_metrics, self.tf_domain_features, self.tf_datamodule, self.tf_datainfos = out

        # get extra_features
        self.extra_features, self.datainfos = self.get_extra_features(
            self.datainfos,
            self.datamodule,
            self.domain_features)
        self.nodes_dist = self.datainfos.nodes_dist
        if self.args.transfer:
            self.tf_extra_features, self.tf_datainfos = self.get_extra_features(
                self.tf_datainfos,
                self.tf_datamodule,
                self.tf_domain_features)
            self.tf_nodes_dist = self.tf_datainfos.nodes_dist
        else:
            self.tf_extra_features = self.tf_datainfos = self.tf_nodes_dist = None

        # get final stats
        init_ds = self.datamodule.inner
        self.mean_final = PlaceHolder(
            X=torch.ones(1).to(self.device)*0,
            E=torch.ones(2).to(self.device)*0.5*0,
            y=None
        )
        # import pdb; pdb.set_trace()
        self.var_final = PlaceHolder(
            X=torch.ones(self.datainfos.num_node_types, device=self.device)*1e0,
            E=torch.ones(self.datainfos.num_edge_types, device=self.device)*1e0,
            y=None
        )
        self.std_final = PlaceHolder(
            X = torch.sqrt(self.var_final.X),
            E = torch.sqrt(self.var_final.E),
            y=None
        )

        self.decart_mean_final = PlaceHolder(
            X=torch.ones(1).to(self.device),
            E=torch.ones(2).to(self.device)*0.5,
            y=None
        )
        # # import pdb; pdb.set_trace()
        # self.decart_var_final = PlaceHolder(
        #     X=torch.ones(self.datainfos.num_node_types, device=self.device)*1e0,
        #     E=torch.ones(self.datainfos.num_edge_types, device=self.device)*1e0,
        #     y=None
        # )
        # self.decart_std_final = PlaceHolder(
        #     X = torch.sqrt(self.var_final.X),
        #     E = torch.sqrt(self.var_final.E),
        #     y=None
        # )

        # get dataloaders from datamodules
        # what is the difference between save_init—dl and cache_init_dl?
        self.save_init_dl = pygloader.DataLoader(
            init_ds, batch_size=self.args.plot_npar, shuffle=True)  # , **self.kwargs)
        self.cache_init_dl = pygloader.DataLoader(
            init_ds, batch_size=self.args.cache_npar, shuffle=True)  # , **self.kwargs)
        (self.cache_init_dl, self.save_init_dl) = self.accelerator.prepare(
            self.cache_init_dl, self.save_init_dl)
        self.cache_init_dl = repeater(self.cache_init_dl)
        self.save_init_dl = repeater(self.save_init_dl)

        if self.args.transfer:
            final_ds = self.tf_datamodule.inner
            self.save_final_dl = pygloader.DataLoader(
                final_ds, batch_size=self.args.plot_npar, shuffle=True)  # , **self.kwargs)
            self.cache_final_dl = pygloader.DataLoader(
                final_ds, batch_size=self.args.cache_npar, shuffle=True)  # , **self.kwargs)
            (self.cache_final_dl, self.save_final_dl) = self.accelerator.prepare(
                self.cache_final_dl, self.save_final_dl)
            self.cache_final_dl = repeater(self.cache_final_dl)
            self.save_final_dl = repeater(self.save_final_dl)
        else:
            self.cache_final_dl = None
            self.save_final = None

        # get all type of dataloader (currently only for the initial dataset)
        init_train_dl = pygloader.DataLoader(init_ds, batch_size=self.args.plot_npar, shuffle=False)
        print('creating the val dataloader')
        # init_val_ds, _, _, _ = get_datasets(self.args, split='val')
        init_val_ds = self.datamodule.dataloaders['val'].dataset
        init_test_dl = pygloader.DataLoader(init_val_ds, batch_size=self.args.plot_npar, shuffle=False)
        print('creating the test dataloader')
        # init_test_ds, _, _, _ = get_datasets(self.args, split='test')
        init_test_ds = self.datamodule.dataloaders['test'].dataset
        init_val_dl = pygloader.DataLoader(init_test_ds, batch_size=self.args.plot_npar, shuffle=False)

        return {'train': init_train_dl, 'val': init_val_dl, 'test': init_test_dl}

    def new_cacheloader(self, forward_or_backward, n, use_ema=True):

        sample_direction = 'f' if forward_or_backward == 'b' else 'b'
        if use_ema:
            sample_net = self.ema_helpers[sample_direction].ema_copy(
                self.net[sample_direction])
        else:
            sample_net = self.net[sample_direction]

        if forward_or_backward == 'b':
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader('b',
                                 sample_net,
                                 self.cache_init_dl,
                                 self.args.num_cache_batches,
                                 self.langevin, n,
                                 mean=self.mean_final,
                                 std=self.std_final,
                                 batch_size=self.args.cache_npar,
                                 device=self.device,
                                 dataloader_f=self.cache_final_dl,
                                 transfer=self.args.transfer,
                                 graph=self.graph,
                                 nodes_dist=self.nodes_dist,
                                 dataset_infos=self.datainfos,
                                 visualization_tools=self.visualization_tools,
                                 visualize=True)
        else:  # forward
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader('f',
                                 sample_net,
                                 None,
                                 self.args.num_cache_batches,
                                 self.langevin, n,
                                 mean=self.mean_final,
                                 std=self.std_final,
                                 batch_size=self.args.cache_npar,
                                 device=self.device,
                                 dataloader_f=self.cache_final_dl,
                                 transfer=self.args.transfer,
                                 graph=self.graph,
                                 nodes_dist=self.nodes_dist,
                                 dataset_infos=self.datainfos,
                                 visualization_tools=self.visualization_tools,
                                 visualize=True)

        new_dl = pygloader.DataLoader(new_dl, batch_size=self.batch_size, shuffle=True)

        new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)

        return new_dl

    def save_step(self, i, n, fb):
        '''
        Step for sampling and for saving
        '''
        if self.accelerator.is_local_main_process:
            # save checkpoint
            if ((i % self.stride == 0) or ( i == self.num_iter)) and (i > 0):  #  or (i % self.stride == 1))

                if self.args.ema:
                    sample_net = self.ema_helpers[fb].ema_copy(self.net[fb])
                else:
                    sample_net = self.net[fb]

                name_net = 'net' + '_' + fb + '_' + \
                    str(n) + "_" + str(i) + '.ckpt'
                name_net_ckpt = './checkpoints/' + name_net

                if self.args.dataparallel:
                    torch.save(self.net[fb].module.state_dict(), name_net_ckpt)
                else:
                    torch.save(self.net[fb].state_dict(), name_net_ckpt)

                if self.args.ema:
                    name_net = 'sample_net' + '_' + fb + \
                        '_' + str(n) + "_" + str(i) + '.ckpt'
                    name_net_ckpt = './checkpoints/' + name_net
                    if self.args.dataparallel:
                        torch.save(sample_net.module.state_dict(),
                                   name_net_ckpt)
                    else:
                        torch.save(sample_net.state_dict(), name_net_ckpt)

            # get val and test results
            if ((i % (2*self.stride) == 0) or ( i == self.num_iter)) and (i > 0):
                with torch.no_grad():
                    self.set_seed(seed=0 + self.accelerator.process_index)
                    if fb == 'f' or self.args.transfer:
                        batch = next(self.save_init_dl)
                        batch, node_mask = utils.data_to_dense(batch, self.max_n_nodes)
                        batch = batch.minus(self.decart_mean_final)
                        batch = batch.scale(4)
                        n_nodes = node_mask.sum(-1)
                        batch.X = torch.zeros_like(batch.X, device=batch.X.device)
                        batch.E = batch.E[:,:,:,-1].unsqueeze(-1)
                        batch = batch.mask(node_mask)
                    else:
                        batch_size = self.args.plot_npar
                        n_nodes = self.nodes_dist.sample_n(batch_size, self.device)
                        batch = utils.PlaceHolder(
                            X=torch.zeros(batch_size,
                                        self.max_n_nodes,
                                        len(self.datainfos.node_types)).to(self.device),
                            E=torch.randn(batch_size,
                                        self.max_n_nodes,
                                        self.max_n_nodes,
                                        len(self.datainfos.node_types)).to(self.device),
                            y=None, charge=None
                        )
                        batch.E = utils.symmetize_edge_matrix(batch.E)
                        arange = (
                            torch.arange(self.max_n_nodes, device=self.device).unsqueeze(0).expand(batch_size, -1)
                        )
                        node_mask = arange < n_nodes.unsqueeze(1)
                        batch.mask(node_mask)

                    x_tot, out, steps_expanded = self.langevin.record_langevin_seq(
                        sample_net, batch, node_mask=node_mask, ipf_it=n, sample=True, fb=fb)

                # # add metrics for graphs
                # # generated_graphs need to be resized, delete the channel dimention
                # generated_graphs = x_tot_plot.cpu().numpy()  # (n_steps, batch_size, n_channels, n_nodes, n_nodes)
                # generated_graphs = generated_graphs[0, :, 0, :, :]  # (batch_size, n_nodes, n_nodes)
                # generated_list = []
                # for l in range(len(generated_graphs)):
                #     n_node = n_nodes[l]
                #     generated_list.append(generated_graphs[l][:n_node, :n_node])

                to_plot = x_tot.get_data(-1, dim=1).collapse(thres=self.args.thres)

                n_nodes = to_plot.node_mask.sum(-1)
                X = to_plot.X
                E = to_plot.E
                generated_list = []
                for l in range(X.shape[0]):
                    cur_n = n_nodes[l]
                    atom_types = X[l, :cur_n].cpu()
                    edge_types = E[l, :cur_n, :cur_n].cpu()
                    generated_list.append([atom_types, edge_types])

                print('visualizing graphs...')
                current_path = os.getcwd()
                result_path = os.path.join(
                    current_path,
                    f"im/step{str(i)}_iter{str(n)}_{fb}/",
                )

                self.visualization_tools.visualize(
                    result_path,
                    to_plot,
                    atom_decoder=None,
                    num_graphs_to_visualize=to_plot.X.shape[0]
                )

                print('Visualizing chains...')

                result_path = os.path.join(current_path, f'predict_chains_{fb}/ipf{n}_step{i}')

                chain = x_tot.copy()
                chain.X = torch.concatenate((batch.X.unsqueeze(1), chain.X),dim=1)
                chain.E = torch.concatenate((batch.E.unsqueeze(1), chain.E),dim=1)
                _ = self.visualization_tools.visualize_chains(result_path,
                                                                chains=chain,
                                                                num_nodes=n_nodes,
                                                                local_rank=0,
                                                                num_chains_to_visualize=2,
                                                                fb=fb
                )
                mean_val = chain.E.mean(-1).mean(-1).mean(-1).mean(0).cpu().numpy()
                np.save(f'{fb}_{n}_mean.npy', mean_val)
                dataa = [[x, y] for (x, y) in zip(np.arange(chain.E.shape[1]), mean_val)]
                if mean_val.max() > 10 or mean_val.min() < -10:
                    import pdb; pdb.set_trace()
                table = wandb.Table(data=dataa, columns=["steps", "mean"])
                wandb.log({f'{fb}_{n}_mean_pred': wandb.plot.line(table, "steps", "mean", title=f'{fb}_{n}_mean_pred')})

                if self.test_sampling_metrics is not None:
                    test_to_log = self.test_sampling_metrics.compute_all_metrics(
                        generated_list, current_epoch=0, local_rank=0, fb=fb, i=np.round(i/(self.num_iter+1),2)
                    )
                    # save results for testing
                    print('saving results for testing')
                    current_path = os.getcwd()
                    res_path = os.path.join(
                        current_path,
                        f"test_epoch{self.current_epoch}.json",
                    )

                    with open(res_path, 'w') as file:
                        # Convert the dictionary to a JSON string and write it to the file
                        json.dump(test_to_log, file)

                elif self.val_sampling_metrics is not None:
                    val_to_log = self.val_sampling_metrics.compute_all_metrics(
                        generated_list, current_epoch=0, local_rank=0, fb=fb, i=np.round(i/(self.num_iter+1),2)
                    )
                    # save results for testing
                    print('saving results for testing')
                    current_path = os.getcwd()
                    res_path = os.path.join(
                        current_path,
                        f"val_epoch{self.current_epoch}.json",
                    )

                    with open(res_path, 'w') as file:
                        # Convert the dictionary to a JSON string and write it to the file
                        json.dump(val_to_log, file)

                    # if wandb.run:
                    #     val_to_log = {f"sampling/{k}_{fb}": val_to_log[k] for k in val_to_log.keys()}
                    #     wandb.log(val_to_log, commit=True, step=n)

                # self.plotter(batch, x_tot_plot, i, n, fb)

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        torch.cuda.empty_cache()


class IPFSequential(IPFBase):

    def __init__(self, args):
        super().__init__(args)

    def ipf_step(self, forward_or_backward, n):
        if n == '3':
            import pdb; pdb.set_trace()

        new_dl = self.new_cacheloader(forward_or_backward, n, self.args.ema)

        if not self.args.use_prev_net:
            self.build_models(forward_or_backward)
            self.update_ema(forward_or_backward)

        self.build_optimizers()
        self.accelerate(forward_or_backward)

        cur_num_iter = self.num_iter + 1
        # if forward_or_backward == 'f':
        #     cur_num_iter = cur_num_iter * 5

        for i in tqdm(range(cur_num_iter)):
            '''
            training step
            '''
            self.set_seed(seed=n*self.num_iter + i)
            x, out, steps_expanded = next(new_dl)
            x = PlaceHolder(X=x[0], E=x[1], y=x[2], charge=x[3], n_nodes=x[4])
            out = PlaceHolder(X=out[0], E=out[1], y=out[2], charge=out[3], n_nodes=out[4])
            # import pdb; pdb.set_trace()
            eval_steps = self.T - steps_expanded
            # import pdb; pdb.set_trace()
            pred = self.forward_graph(self.net[forward_or_backward], x, eval_steps)

            # if forward_or_backward == 'f':
            #     new_x = x.copy()
            #     new_x.E[0,:,:,0] = torch.Tensor([[0,2,2,2],[2,0,-2,-2], [2,-2,0,-2], [-2,-2,-2,0]]).to(new_x.E.device)
            #     new_eval_steps = eval_steps.clone()
            #     new_eval_steps[0] = torch.tensor(1e-5).to(new_x.E.device)
            #     new_pred = self.forward_graph(self.net[forward_or_backward], new_x, new_eval_steps)
            #     print(i, new_pred.E[0,:,:,0], x.E[0,:,:,0], new_eval_steps[0])

            if self.args.mean_match:
                pred = pred.minus(x)

            # if forward_or_backward == 'f' and i==500:
            #     import pdb; pdb.set_trace()

            node_loss, edge_loss, loss = self.compute_loss(pred, out)

            if wandb.run:
                wandb.log({"num_ipf": n}, commit=True)
                if forward_or_backward == 'f':
                    wandb.log({"forward_num_iter": n * self.num_iter + i + 1}, commit=True)
                else:
                    wandb.log({"backward_num_iter": n * self.num_iter + i + 1}, commit=True)
                wandb.log(
                    {
                        f"train/loss_{forward_or_backward}_{n}": loss.detach().cpu().numpy().item(),
                        f"train/loss_edge_{forward_or_backward}_{n}": edge_loss.detach().cpu().numpy().item(),
                        f"train/loss_node_{forward_or_backward}_{n}": node_loss.detach().cpu().numpy().item(),
                    },
                    commit=False,
                )

            # loss.backward()
            self.accelerator.backward(loss)

            if self.grad_clipping:
                clipping_param = self.args.grad_clip
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net[forward_or_backward].parameters(), clipping_param)
            else:
                total_norm = 0.

            if (i % self.stride_log == 0) and (i > 0):
                self.logger.log_metrics({'forward_or_backward': forward_or_backward,
                                         'loss': loss,
                                         'grad_norm': total_norm},
                                        # step=i})
                                        step=i+self.num_iter*n)

            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad()
            if self.args.ema:
                self.ema_helpers[forward_or_backward].update(
                    self.net[forward_or_backward])

            self.save_step(i, n, forward_or_backward)

            if (i % self.args.cache_refresh_stride == 0) and (i > 0):
                new_dl = None
                torch.cuda.empty_cache()
                new_dl = self.new_cacheloader(
                    forward_or_backward, n, self.args.ema)

        new_dl = None
        self.clear()

    def compute_loss(self, pred, out):
        node_loss = self.args.model.lambda_train[0] * F.mse_loss(pred.X, out.X) * 0
        edge_loss = self.args.model.lambda_train[1] * F.mse_loss(pred.E[:,:,:,-1], out.E[:,:,:,-1])
        # print('loss',  pred.E[0,0,1].detach(), out.E[0,0,1].detach(), pred.E[0,0,1]-out.E[0,0,1])
        loss = node_loss + edge_loss
        # The code is calculating the total loss by adding the node loss and edge loss together.
        # loss = edge_loss
        if pred.charge.numel() > 0:
            loss = loss + self.args.model.lambda_train[0] * F.mse_loss(pred.E, out.E)

        return node_loss, edge_loss, loss

    def train(self):
        print('Training...')
        # INITIAL FORWARD PASS
        # if self.accelerator.is_local_main_process:
        #     init_sample = next(self.save_init_dl)  # under the form of pyg-data
        #     init_sample, node_mask = utils.data_to_dense(init_sample, self.max_n_nodes)
        #     x_tot, _, _ = self.langevin.record_init_langevin(init_sample, node_mask)
        #     # TODO: add visualization here, but visualization might be mitigated to the validation part as well
        #     torch.cuda.empty_cache()

        for n in range(self.checkpoint_it, self.n_ipf+1):
            print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_step('f', n)
            else:
                self.ipf_step('b', n)
                self.ipf_step('f', n)

    # def forward_graph(self, net, z_t, t):
    #     # step 1: calculate extra features
    #     assert z_t.node_mask is not None
    #     model_input = z_t.copy()
    #     model_input.y = torch.hstack(
    #         (z_t.y, t)
    #     ).float()

    #     return net(model_input)


    def forward_graph(self, net, z_t, t):
        # step 1: calculate extra features
        assert z_t.node_mask is not None
        # z_t.X = z_t.X / (z_t.X.abs().sum(-1).unsqueeze(-1)+1e-6)
        # z_t.E = z_t.E / (z_t.E.abs().sum(-1).unsqueeze(-1)+1e-6)
        z_t = z_t.clip(3)

        z_t_discrete = z_t.collapse()
        model_input = z_t.copy()
        # z_t.E = (z_t.E > 0.5).long()
        extra_features, _, _ = self.extra_features(z_t_discrete)
        # import pdb; pdb.set_trace()
        z_t_discrete.E = z_t_discrete.E.unsqueeze(-1)
        z_t_discrete.X = z_t_discrete.X.unsqueeze(-1)
        extra_domain_features = self.domain_features(z_t_discrete)
        # extra_features = self.extra_features(z_t_discrete)
        # extra_domain_features = self.domain_features(z_t_discrete)

        # # step 2: forward to the langevin process
        # # Need to copy to preserve dimensions in transition to z_{t-1} in sampling (prevent changing dimensions of z_t

        model_input.X = torch.cat(
            (z_t.X, extra_features.X, extra_domain_features.X), dim=2
        ).float()
        model_input.E = torch.cat(
            (z_t.E, extra_features.E, extra_domain_features.E), dim=3
        ).float()
        model_input.y = torch.hstack(
            (z_t.y, extra_features.y, extra_domain_features.y, t)
        ).float()
        
        # print('max input', model_input.E.max(), model_input.E.min())
        res = net(model_input)
        # res = res.clip(3)
        # print('max result', res.E.max(), res.E.min())

        return res
