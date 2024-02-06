
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.autograd.profiler as profiler
from torch.utils.data import WeightedRandomSampler
from accelerate import Accelerator, DistributedType

from .ema import EMAHelper
from . import repeater
from .config_getters import get_models, get_optimizers, get_datasets, get_plotter, get_logger
from ..utils import get_masks
from ..data import CacheLoader
from ..langevin import Langevin
from ..metrics.sampling_metrics import SamplingMetrics



def setup_wandb(cfg):
    kwargs = {
        "name": cfg.Dataset,
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

        # get models
        print('building models')
        self.build_models()
        self.build_ema()

        # get optims
        self.build_optimizers()

        # get loggers
        self.logger = self.get_logger()
        self.save_logger = self.get_logger('plot_logs')

        # get data
        print('creating dataloaders')
        dataloaders = self.build_dataloaders()

        # create metrics for graph dataset
        if self.graph:
            print('creating metrics for graph dataset')
            # create the training/test/val dataloader
            self.val_sampling_metrics = SamplingMetrics(
                dataset_infos=None, test=False, dataloaders=dataloaders
            )
            self.test_sampling_metrics = SamplingMetrics(
                dataset_infos=None, test=True, dataloaders=dataloaders
            )

        # langevin
        if self.args.weight_distrib:
            alpha = self.args.weight_distrib_alpha
            prob_vec = (1 + alpha) * torch.sum(gammas) - \
                torch.cumsum(gammas, 0)
        else:
            prob_vec = gammas * 0 + 1
        time_sampler = torch.distributions.categorical.Categorical(prob_vec)

        # print('next batch debug')
        batch = next(self.save_init_dl)[0]  # TODO: this next operation is never done
        shape = batch[0].shape
        self.shape = shape
        # self.shape = shape = (20, 20)
        print('creating Langevin')
        self.langevin = Langevin(self.num_steps, shape, gammas,
                                 time_sampler, device=self.device,
                                 mean_final=self.mean_final, var_final=self.var_final,
                                 mean_match=self.args.mean_match, graph=self.graph)

        # checkpoint
        date = str(datetime.datetime.now())[0:10]
        self.name_all = date

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
            if not os.path.exists('./gif'):
                os.mkdir('./gif')
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
        net_f, net_b = get_models(self.args)

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
                sample_net_f, sample_net_b = get_models(self.args)

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

    def build_dataloaders(self):
        init_ds, final_ds, mean_final, var_final = get_datasets(self.args)

        # var_final and mean_final are ... ?
        self.mean_final = mean_final.to(self.device)
        self.var_final = var_final.to(self.device)
        self.std_final = torch.sqrt(var_final).to(self.device)

        self.node_dist = None
        if self.graph:
            self.node_dist = init_ds.node_dist

        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[
                           1][0] + worker_id + self.accelerator.process_index)

        self.kwargs = {"num_workers": self.args.num_workers,
                       "pin_memory": self.args.pin_memory,
                       "worker_init_fn": worker_init_fn,
                       "drop_last": True}

        # get plotter, gifs etc.
        # what is the difference between save_init—dl and cache_init_dl?
        self.save_init_dl = DataLoader(
            init_ds, batch_size=self.args.plot_npar, shuffle=True, **self.kwargs)
        self.cache_init_dl = DataLoader(
            init_ds, batch_size=self.args.cache_npar, shuffle=True, **self.kwargs)
        (self.cache_init_dl, self.save_init_dl) = self.accelerator.prepare(
            self.cache_init_dl, self.save_init_dl)
        self.cache_init_dl = repeater(self.cache_init_dl)
        self.save_init_dl = repeater(self.save_init_dl)

        if self.args.transfer:
            self.save_final_dl = DataLoader(
                final_ds, batch_size=self.args.plot_npar, shuffle=True, **self.kwargs)
            self.cache_final_dl = DataLoader(
                final_ds, batch_size=self.args.cache_npar, shuffle=True, **self.kwargs)
            (self.cache_final_dl, self.save_final_dl) = self.accelerator.prepare(
                self.cache_final_dl, self.save_final_dl)
            self.cache_final_dl = repeater(self.cache_final_dl)
            self.save_final_dl = repeater(self.save_final_dl)
        else:
            self.cache_final_dl = None
            self.save_final = None

        # get all type of dataloader
        init_train_dl = DataLoader(init_ds, batch_size=self.args.plot_npar, shuffle=False)
        print('creating the val dataloader')
        init_val_ds, _, _, _ = get_datasets(self.args, split='val')
        init_test_dl = DataLoader(init_val_ds, batch_size=self.args.plot_npar, shuffle=False)
        print('creating the test dataloader')
        init_test_ds, _, _, _ = get_datasets(self.args, split='test')
        init_val_dl = DataLoader(init_test_ds, batch_size=self.args.plot_npar, shuffle=False)
        
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
                                 mean=None,
                                 std=None,
                                 batch_size=self.args.cache_npar,
                                 device=self.device,
                                 dataloader_f=self.cache_final_dl,
                                 transfer=self.args.transfer,
                                 graph=self.graph,
                                 node_dist=self.node_dist)

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
                                 node_dist=self.node_dist)

        new_dl = DataLoader(new_dl, batch_size=self.batch_size)

        new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)
        return new_dl

    def train(self):
        pass

    def save_step(self, i, n, fb):
        '''
        Step for sampling and for saving
        '''
        if self.accelerator.is_local_main_process:
            if ((i % self.stride == 0) or (i % self.stride == 1)) and (i > 0):

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

                with torch.no_grad():
                    self.set_seed(seed=0 + self.accelerator.process_index)
                    if fb == 'f':
                        batch = next(self.save_init_dl)
                        if self.graph:
                            batch, n_nodes = batch
                            batch = (batch.to(self.device), n_nodes.to(self.device))
                        else:
                            batch = batch[0].to(self.device)
                    elif self.args.transfer:
                        batch = next(self.save_final_dl)
                        if self.graph:
                            batch, n_nodes = batch
                            batch = (batch.to(self.device), n_nodes.to(self.device))
                        else:
                            batch = batch[0].to(self.device)
                    else:
                        if self.graph:
                            batch_size = self.args.plot_npar
                            n_nodes = self.node_dist.sample_n(batch_size, self.device)
                            _, edge_mask = get_masks(
                                n_nodes, self.node_dist.max_n_nodes, batch_size, self.device)
                        batch = self.mean_final + self.std_final * \
                            torch.randn(
                                (self.args.plot_npar, *self.shape), device=self.device)
                        if self.graph:
                            batch = batch * edge_mask.unsqueeze(1)
                            batch = (batch, n_nodes.to(self.device))

                    x_tot, out, steps_expanded = self.langevin.record_langevin_seq(
                        sample_net, batch, ipf_it=n, sample=True)
                    if self.graph:
                        x_tot = x_tot[0]
                        batch = batch[0]

                    shape_len = len(x_tot.shape)
                    x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
                    
                    # threshouding operation for graph
                    if self.graph:
                        x_tot = (x_tot >= 0.5).float()
                    x_tot_plot = x_tot.detach()  # .cpu().numpy()

                init_x = batch.detach().cpu().numpy()
                final_x = x_tot_plot[-1].detach().cpu().numpy()
                std_final = np.std(final_x)
                std_init = np.std(init_x)
                mean_final = np.mean(final_x)
                mean_init = np.mean(init_x)

                print('Initial variance: ' + str(std_init ** 2))
                print('Final variance: ' + str(std_final ** 2))

                self.save_logger.log_metrics({'FB': fb,
                                              'init_var': std_init**2, 'final_var': std_final**2,
                                              'mean_init': mean_init, 'mean_final': mean_final,
                                              'T': self.T})
                
                # add metrics for graphs
                if self.graph:
                    # generated_graphs need to be resized, delete the channel dimention
                    generated_graphs = x_tot_plot.cpu().numpy()  # (n_steps, batch_size, n_channels, n_nodes, n_nodes)
                    generated_graphs = generated_graphs[0, :, 0, :, :]  # (batch_size, n_nodes, n_nodes)
                    generated_list = []
                    for i in range(len(generated_graphs)):
                        n = n_nodes[i]
                        generated_list.append(generated_graphs[i][:n, :n])
                    if self.test_sampling_metrics is not None:
                        test_to_log = self.test_sampling_metrics.compute_all_metrics(
                            generated_graphs, current_epoch=0, local_rank=0
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

                        if wandb.run:
                            test_to_log = {f"val/{k}": test_to_log[k] for k in test_to_log.keys()}
                            wandb.log(test_to_log, commit=False)

                    elif self.val_sampling_metrics is not None:
                        val_to_log = self.val_sampling_metrics.compute_all_metrics(
                            generated_graphs, current_epoch=0, local_rank=0
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


                        if wandb.run:
                            val_to_log = {f"val/{k}": val_to_log[k] for k in val_to_log.keys()}
                            wandb.log(val_to_log, commit=False)

                self.plotter(batch, x_tot_plot, i, n, fb)

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
        new_dl = None
        new_dl = self.new_cacheloader(forward_or_backward, n, self.args.ema)

        if not self.args.use_prev_net:
            self.build_models(forward_or_backward)
            self.update_ema(forward_or_backward)

        self.build_optimizers()
        self.accelerate(forward_or_backward)

        for i in tqdm(range(self.num_iter+1)):
            '''
            training step
            '''
            self.set_seed(seed=n*self.num_iter + i)
            x, out, steps_expanded = next(new_dl)
            # get masks for graph
            if self.graph:
                # import pdb; pdb.set_trace()
                # nbr of nodes need to be passed as an argument
                x, n_nodes = x
                bs = x.shape[0]
                max_n = x.shape[-1]  # for categorical graphs, this need to be changes
                node_mask, edge_mask = get_masks(n_nodes, max_n, bs, self.device)
                x = x * edge_mask.unsqueeze(1)
                out = out * edge_mask.unsqueeze(1)

            x = x.to(self.device)
            out = out.to(self.device)
            steps_expanded = steps_expanded.to(self.device)
            # eval_steps = self.num_steps - 1 - steps_expanded
            eval_steps = self.T - steps_expanded

            if self.args.mean_match:
                pred = self.net[forward_or_backward](
                    x, eval_steps) - x
            else:
                pred = self.net[forward_or_backward](x, eval_steps)
            # mask pred
            if self.graph:
                pred = pred * edge_mask.unsqueeze(1)

            loss = F.mse_loss(pred, out)

            if wandb.run:
                wandb.log({"num_ipf": n}, commit=True)
                wandb.log({"num_iter": n * self.num_iter + i + 1}, commit=True)
                wandb.log(
                    {
                        f"train/loss_{forward_or_backward}": loss.detach().cpu().numpy().item(),
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
                                         'grad_norm': total_norm}, step=i+self.num_iter*n)

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

    def train(self):
        print('Training...')
        # INITIAL FORWARD PASS
        if self.accelerator.is_local_main_process:
            init_sample = next(self.save_init_dl)
            if self.graph:
                init_sample, n_nodes = init_sample
                init_sample = (init_sample.to(self.device), n_nodes.to(self.device))
            else:
                init_sample = init_sample[0]
                init_sample = init_sample.to(self.device)

            x_tot, _, _ = self.langevin.record_init_langevin(init_sample)
            if self.graph:
                x_tot = x_tot[0]
                init_sample = init_sample[0]

            shape_len = len(x_tot.shape)
            # x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
            x_tot_plot = x_tot.detach()
 
            # import pdb; pdb.set_trace()
            self.plotter(init_sample, x_tot_plot, 0, 0, 'f')
            x_tot_plot = None
            x_tot = None
            torch.cuda.empty_cache()

        for n in range(self.checkpoint_it, self.n_ipf+1):
            print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_step('f', n)
            else:
                self.ipf_step('b', n)
                self.ipf_step('f', n)
