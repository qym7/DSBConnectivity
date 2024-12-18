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
from accelerate import Accelerator, DistributedType

# from torch_geometric.loader import DataLoader as pygloader.DataLoader
import torch_geometric.loader as pygloader

from .ema import EMAHelper
from . import repeater
from .config_getters import *  # get_models, get_graph_models, get_optimizers, get_datasets, get_plotter, get_logger
from .. import utils
from ..data import CacheLoader
from ..langevin import Langevin
from ..metrics.sampling_metrics import SamplingMetrics
from ..analysis.visualization import Visualizer
from ..diffusion.extra_features import (
    DummyExtraFeatures,
    ExtraFeatures,
)
from ..analysis.trajectory_metrics import (
    ce_trajectory,
    abs_trajectory,
    accumulated_abs_trajectory,
)


def setup_wandb(cfg):
    kwargs = {
        "name": f"{cfg.name}",
        "project": f"DSB_{cfg.project_name}",
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
        self.cfg = args

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(fp16=False, cpu=args.device == "cpu")
        self.device = self.accelerator.device  # torch.device(args.device)

        # training params
        self.n_ipf = self.cfg.n_ipf
        self.num_steps = self.cfg.num_steps
        self.batch_size = self.cfg.batch_size
        self.num_iter = self.cfg.num_iter
        self.grad_clipping = self.cfg.grad_clipping
        self.grad_clip_value = self.cfg.grad_clip
        self.fast_sampling = self.cfg.fast_sampling
        self.lr = self.cfg.lr

        # sampling (training) step setting
        time_series = torch.linspace(0, 1, self.num_steps + 1).to(self.device)
        if self.cfg.sample.time_distortion == "polydec":
            time_series = time_series * 2 - time_series**2
        elif self.cfg.sample.time_distortion == "polyinc":
            time_series = time_series**2
        elif self.cfg.sample.time_distortion == "cos":
            time_series = (1 - torch.cos(time_series * torch.pi)) / 2
        elif self.cfg.sample.time_distortion == "revcos":
            time_series = (
                2 * time_series - (1 - torch.cos(time_series * torch.pi)) / 2
            )
        gammas = time_series.diff()
        self.T = torch.sum(gammas)
        self.current_epoch = 0

        # get loggers
        self.logger = self.get_logger()
        self.save_logger = self.get_logger("plot_logs")

        # get data
        print("creating dataloaders")
        dataloaders, tf_dataloaders = self.build_datamodules()
        self.max_n_nodes = self.datainfos.max_n_nodes
        self.visualization_tools = Visualizer(dataset_infos=self.datainfos)

        # create metrics for graph dataset
        print("creating metrics for graph dataset")
        # create the training/test/val dataloader

        if self.cfg.transfer:
            self.val_sampling_metrics = SamplingMetrics(
                dataset_infos=self.datainfos,
                test=False,
                dataloaders=dataloaders,
            )
            self.test_sampling_metrics = SamplingMetrics(
                dataset_infos=self.datainfos,
                test=True,
                dataloaders=dataloaders,
            )
            self.tf_val_sampling_metrics = SamplingMetrics(
                dataset_infos=self.tf_datainfos,
                test=False,
                dataloaders=tf_dataloaders,
            )
            self.tf_test_sampling_metrics = SamplingMetrics(
                dataset_infos=self.tf_datainfos,
                test=True,
                dataloaders=tf_dataloaders,
            )
        else:
            self.val_sampling_metrics = SamplingMetrics(
                dataset_infos=self.datainfos,
                test=False,
                dataloaders=dataloaders,
            )
            self.test_sampling_metrics = SamplingMetrics(
                dataset_infos=self.datainfos,
                test=True,
                dataloaders=dataloaders,
            )
            self.tf_val_sampling_metrics = self.tf_test_sampling_metrics = None

        # get models
        print("building models")
        self.build_models()
        self.build_ema()

        # get optims
        self.build_optimizers(n=0)

        # langevin
        if self.cfg.weight_distrib:
            alpha = self.cfg.weight_distrib_alpha
            prob_vec = (1 + alpha) * torch.sum(gammas) - torch.cumsum(gammas, 0)
        else:
            prob_vec = gammas * 0 + 1
        time_sampler = torch.distributions.categorical.Categorical(prob_vec)

        max_n_nodes = self.datainfos.max_n_nodes
        print("creating Langevin")
        if self.cfg.limit_dist == "marginal":
            self.limit_dist = utils.PlaceHolder(
                X=self.datainfos.node_types,
                E=self.datainfos.edge_types,
                y=None,
            )
        if self.cfg.limit_dist == "absorbing":
            dist_X = torch.zeros_like(self.datainfos.node_types)
            dist_X[0] = 1
            dist_E = torch.zeros_like(self.datainfos.edge_types)
            dist_E[0] = 1
            self.limit_dist = utils.PlaceHolder(X=dist_X, E=dist_E, y=None)
        elif self.cfg.limit_dist == "marginal_tf":
            self.limit_dist = utils.PlaceHolder(
                X=self.tf_datainfos.node_types,
                E=self.tf_datainfos.edge_types,
                y=None,
            )
        else:
            self.limit_dist = utils.PlaceHolder(
                X=torch.ones_like(self.datainfos.node_types)
                / len(self.datainfos.node_types),
                E=torch.ones_like(self.datainfos.edge_types)
                / len(self.datainfos.edge_types),
                y=None,
            )

        # add virtual node dimension to the limit_dist
        if self.cfg.virtual_node:
            real_node_ratio = self.tf_datainfos.real_node_ratio
            if self.cfg.limit_dist == "marginal":
                self.limit_dist.X = torch.cat(
                    [
                        1 - torch.ones(1) * real_node_ratio,
                        real_node_ratio * self.limit_dist.X,
                    ]
                )
            elif self.cfg.limit_dist == "marginal_tf":
                self.limit_dist.X = torch.cat(
                    [
                        1 - torch.ones(1) * real_node_ratio,
                        real_node_ratio * self.limit_dist.X,
                    ]
                )
            else:
                self.limit_dist.X = torch.ones(self.limit_dist.X.shape[0] + 1)
            self.limit_dist.X = self.limit_dist.X / self.limit_dist.X.sum()

        self.langevin = Langevin(
            self.num_steps,
            max_n_nodes,
            gammas,
            time_sampler,
            limit_dist=self.limit_dist,
            device=self.device,
            extra_features=self.extra_features,
            domain_features=self.domain_features,
            tf_extra_features=self.tf_extra_features,
            tf_domain_features=self.tf_domain_features,
            virtual_node=self.cfg.virtual_node,
            noise_level=self.cfg.noise_level,
        )

        # checkpoint
        date = str(datetime.datetime.now())[0:10]
        self.name_all = date + self.cfg.project_name

        # run from checkpoint
        self.checkpoint_run = self.cfg.checkpoint_run
        if self.cfg.checkpoint_run:
            self.checkpoint_it = self.cfg.checkpoint_it
            self.checkpoint_pass = self.cfg.checkpoint_pass
        else:
            self.checkpoint_it = 1
            self.checkpoint_pass = "b"

        if self.accelerator.process_index == 0:
            if not os.path.exists("./im"):
                os.mkdir("./im")
            if not os.path.exists("./checkpoints"):
                os.mkdir("./checkpoints")

        self.stride = self.cfg.gif_stride
        self.stride_log = self.cfg.log_stride

    def get_logger(self, name="logs"):
        return get_logger(self.cfg, name)

    def build_models(self, forward_or_backward=None):
        # running network
        net_f, net_b = get_graph_models(self.cfg, self.datainfos)

        if self.cfg.checkpoint_run:
            if "checkpoint_f" in self.cfg:
                net_f.load_state_dict(torch.load(self.cfg.checkpoint_f))
            if "checkpoint_b" in self.cfg:
                net_b.load_state_dict(torch.load(self.cfg.checkpoint_b))

        if self.cfg.dataparallel:
            net_f = torch.nn.DataParallel(net_f)
            net_b = torch.nn.DataParallel(net_b)

        if forward_or_backward is None:
            net_f = net_f.to(self.device)
            net_b = net_b.to(self.device)
            self.net = torch.nn.ModuleDict({"f": net_f, "b": net_b})
        if forward_or_backward == "f":
            net_f = net_f.to(self.device)
            self.net.update({"f": net_f})
        if forward_or_backward == "b":
            net_b = net_b.to(self.device)
            self.net.update({"b": net_b})

    def accelerate(self, forward_or_backward):
        (
            self.net[forward_or_backward],
            self.optimizer[forward_or_backward],
        ) = self.accelerator.prepare(
            self.net[forward_or_backward],
            self.optimizer[forward_or_backward],
        )

    def update_ema(self, forward_or_backward):
        if self.cfg.ema:
            self.ema_helpers[forward_or_backward] = EMAHelper(
                mu=self.cfg.ema_rate, device=self.device
            )
            self.ema_helpers[forward_or_backward].register(
                self.net[forward_or_backward]
            )

    def build_ema(self):
        if self.cfg.ema:
            self.ema_helpers = {}
            self.update_ema("f")
            self.update_ema("b")

            # sample network
            sample_net_f, sample_net_b = get_graph_models(
                self.cfg, self.datainfos
            )

            if len(self.cfg.sample_checkpoint_f) > 1:
                sample_net_f.load_state_dict(
                    torch.load(self.cfg.sample_checkpoint_f)
                )
                if self.cfg.dataparallel:
                    sample_net_f = torch.nn.DataParallel(sample_net_f)
                sample_net_f = sample_net_f.to(self.device)
                self.ema_helpers["f"].register(sample_net_f)
                print('forward model loaded for sampling!')
            if len(self.cfg.sample_checkpoint_b) > 1:
                sample_net_b.load_state_dict(
                    torch.load(self.cfg.sample_checkpoint_b)
                )
                if self.cfg.dataparallel:
                    sample_net_b = torch.nn.DataParallel(sample_net_b)
                sample_net_b = sample_net_b.to(self.device)
                self.ema_helpers["b"].register(sample_net_b)
                print('backward model loaded for sampling!')

    def build_optimizers(self, n=0):
        # lr = self.lr / (n+1)  # decay in learning rate
        lr = self.lr
        optimizer_f, optimizer_b = get_optimizers(
            self.net["f"], self.net["b"], lr, n, self.n_ipf
        )
        optimizer_b = optimizer_b
        optimizer_f = optimizer_f
        self.optimizer = {"f": optimizer_f, "b": optimizer_b}

    def get_final_stats(self, init_ds):
        """
        marginal of the distribution should be the starting point
        this function is not correct due to the usage of graph datasets
        """
        if self.cfg.adaptive_mean:
            NAPPROX = 100
            # TODO: this batch size causes error when it can not be devided by the datasize
            import pdb

            pdb.set_trace()
            vec = next(iter(pygloader.DataLoader(init_ds, batch_size=NAPPROX)))[
                0
            ]
            mean_final = vec.mean()
            mean_final = vec[0] * 0 + mean_final
            var_final = eval(self.cfg.var_final)
            final_ds = None
        elif self.cfg.final_adaptive:
            NAPPROX = 100
            vec = next(iter(pygloader.DataLoader(init_ds, batch_size=NAPPROX)))[
                0
            ]
            mean_final = vec.mean(axis=0)
            var_final = vec.var()
            final_ds = None
        else:
            mean_final = eval(self.cfg.mean_final)
            var_final = eval(self.cfg.var_final)
            final_ds = None

        mean_final = mean_final.to(self.device)
        val_final = var_final.to(self.device)
        std_final = torch.sqrt(var_final).to(self.device)

        return mean_final, val_final, std_final

    def get_extra_features(self, dataset_infos, datamodule, domain_features):
        ef = self.cfg.model.extra_features
        extra_features = (
            ExtraFeatures(
                extra_features_type=self.cfg.model.extra_features,
                dataset_info=dataset_infos,
                rrwp_steps=self.cfg.model.rrwp_steps,
                use_positional=self.cfg.model.positional_encoding,
            )
            if ef is not None
            else DummyExtraFeatures()
        )

        dataset_infos.compute_input_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
            virtual_node=self.cfg.virtual_node,
        )

        return extra_features, dataset_infos

    def build_datamodules(self):
        # get datamodules for train/val/test data
        out = get_both_datamodules(self.cfg)
        (
            self.train_metrics,
            self.domain_features,
            self.datamodule,
            self.datainfos,
            self.tf_train_metrics,
            self.tf_domain_features,
            self.tf_datamodule,
            self.tf_datainfos,
        ) = out

        self.extra_features, self.datainfos = self.get_extra_features(
            self.datainfos,
            self.datamodule,
            self.domain_features,
        )

        self.nodes_dist = self.datainfos.nodes_dist
        if self.cfg.transfer:
            self.tf_extra_features, self.tf_datainfos = self.get_extra_features(
                self.tf_datainfos,
                self.tf_datamodule,
                self.tf_domain_features,
            )
            self.tf_nodes_dist = self.tf_datainfos.nodes_dist
        else:
            self.tf_extra_features = self.tf_datainfos = self.tf_nodes_dist = (
                None
            )
            self.tf_datainfos = self.datainfos

        # get final stats
        init_ds = self.datamodule.inner
        self.save_init_dl = pygloader.DataLoader(
            init_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.cache_init_dl = pygloader.DataLoader(
            init_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        (self.cache_init_dl, self.save_init_dl) = self.accelerator.prepare(
            self.cache_init_dl, self.save_init_dl
        )
        self.cache_init_dl = repeater(self.cache_init_dl)
        self.save_init_dl = repeater(self.save_init_dl)

        # for test graphs
        init_ds_test = self.datamodule.dataloaders["test"].dataset

        print(
            "creating the test dataloader with",
            len(init_ds_test),
            "graphs",
        )
        self.save_init_dl_test = pygloader.DataLoader(
            init_ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        self.cache_init_dl_test = pygloader.DataLoader(
            init_ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        (self.cache_init_dl_test, self.save_init_dl_test) = (
            self.accelerator.prepare(
                self.cache_init_dl_test, self.save_init_dl_test
            )
        )
        self.cache_init_dl_test = repeater(self.cache_init_dl_test)
        self.save_init_dl_test = repeater(self.save_init_dl_test)

        # for validation graphs
        init_ds_val = self.datamodule.dataloaders["val"].dataset
        print(
            "creating the validation dataloader with",
            len(init_ds_val),
            "graphs",
        )
        self.save_init_dl_val = pygloader.DataLoader(
            init_ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        self.cache_init_dl_val = pygloader.DataLoader(
            init_ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        (self.cache_init_dl_val, self.save_init_dl_val) = (
            self.accelerator.prepare(
                self.cache_init_dl_val, self.save_init_dl_val
            )
        )
        self.cache_init_dl_val = repeater(self.cache_init_dl_val)
        self.save_init_dl_val = repeater(self.save_init_dl_val)

        # get all type of dataloader (currently only for the initial dataset)
        print("creating the train dataloader")
        init_train_dl = pygloader.DataLoader(
            init_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print("creating the val dataloader")
        init_val_ds = self.datamodule.dataloaders["val"].dataset
        init_val_dl = pygloader.DataLoader(
            init_val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        print("creating the test dataloader")
        init_test_ds = self.datamodule.dataloaders["test"].dataset
        init_test_dl = pygloader.DataLoader(
            init_test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )

        init_loaders = {
            "train": init_train_dl,
            "val": init_val_dl,
            "test": init_test_dl,
        }

        if self.cfg.transfer:
            final_ds = self.tf_datamodule.inner
            self.save_final_dl = pygloader.DataLoader(
                final_ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                drop_last=True,
            )
            self.cache_final_dl = pygloader.DataLoader(
                final_ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                drop_last=True,
            )
            (self.cache_final_dl, self.save_final_dl) = (
                self.accelerator.prepare(
                    self.cache_final_dl, self.save_final_dl
                )
            )
            self.cache_final_dl = repeater(self.cache_final_dl)
            self.save_final_dl = repeater(self.save_final_dl)

            # for test graphs
            final_test_ds = self.tf_datamodule.testing
            self.save_final_dl_test = pygloader.DataLoader(
                final_test_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
            )
            self.cache_final_dl_test = pygloader.DataLoader(
                final_test_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
            )
            (
                self.cache_final_dl_test,
                self.save_final_dl_test,
            ) = self.accelerator.prepare(
                self.cache_final_dl_test,
                self.save_final_dl_test,
            )
            self.cache_final_dl_test = repeater(self.cache_final_dl_test)
            self.save_final_dl_test = repeater(self.save_final_dl_test)

            # for validation graphs
            final_val_ds = self.tf_datamodule.validating
            self.save_final_dl_val = pygloader.DataLoader(
                final_val_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
            )
            self.cache_final_dl_val = pygloader.DataLoader(
                final_val_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
            )
            (
                self.cache_final_dl_val,
                self.save_final_dl_val,
            ) = self.accelerator.prepare(
                self.cache_final_dl_val, self.save_final_dl_val
            )
            self.cache_final_dl_val = repeater(self.cache_final_dl_val)
            self.save_final_dl_val = repeater(self.save_final_dl_val)

            # get all type of dataloader (currently only for the initial dataset)
            print("creating the train dataloader")
            final_train_dl = pygloader.DataLoader(
                final_ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                drop_last=True,
            )
            print("creating the val dataloader")
            final_val_dl = pygloader.DataLoader(
                final_val_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
            )
            print("creating the test dataloader")
            final_test_dl = pygloader.DataLoader(
                final_test_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
            )
            final_loaders = {
                "train": final_train_dl,
                "val": final_val_dl,
                "test": final_test_dl,
            }
        else:
            self.cache_final_dl = None
            self.save_final = None
            final_loaders = None

        return init_loaders, final_loaders

    def new_cacheloader(self, forward_or_backward, n, use_ema=True):
        sample_direction = "f" if forward_or_backward == "b" else "b"
        if use_ema:
            sample_net = self.ema_helpers[sample_direction].ema_copy(
                self.net[sample_direction]
            )
        else:
            sample_net = self.net[sample_direction]

        if forward_or_backward == "b":
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader(
                "b",
                sample_net,
                self.cache_init_dl,
                self.cfg.num_cache_batches,
                self.langevin,
                n,
                limit_dist=self.limit_dist,
                batch_size=self.cfg.batch_size,
                rand_time=self.cfg.rand_time,
                device=self.device,
                dataloader_f=self.cache_final_dl,
                transfer=self.cfg.transfer,
                nodes_dist=self.nodes_dist,
                dataset_infos=self.datainfos,
                visualization_tools=self.visualization_tools,
                virtual_node=self.cfg.virtual_node,
            )

        else:  # forward
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader(
                "f",
                sample_net,
                None,
                self.cfg.num_cache_batches,
                self.langevin,
                n,
                limit_dist=self.limit_dist,
                batch_size=self.cfg.batch_size,
                rand_time=self.cfg.rand_time,
                device=self.device,
                dataloader_f=self.cache_final_dl,
                transfer=self.cfg.transfer,
                nodes_dist=self.nodes_dist,
                dataset_infos=self.datainfos,
                visualization_tools=self.visualization_tools,
                virtual_node=self.cfg.virtual_node,
            )

        new_dl = pygloader.DataLoader(
            new_dl, batch_size=self.batch_size, shuffle=True
        )

        new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)

        return new_dl

    def generate_graphs(self, fb, sample_net, n, test=False):
        if not test:
            samples_to_generate = self.cfg.samples_to_generate
            chains_to_save = self.cfg.chains_to_save
        else:
            samples_to_generate = self.cfg.final_samples_to_generate
            chains_to_save = self.cfg.final_chains_to_save

        samples = []
        init_samples = []
        batches = []
        all_n_nodes = []

        X_nbr_changes = 0
        E_nbr_changes = 0
        X_ratio_changes = 0
        E_ratio_changes = 0

        i = 0
        samples_to_return = samples_to_generate

        while samples_to_generate > 0:
            if fb == "f" or self.cfg.transfer:
                if test:
                    loader = (
                        self.save_init_dl_test
                        if fb == "f"
                        else self.save_final_dl_test
                    )
                    batch = next(loader)
                    batch, node_mask = utils.data_to_dense(
                        batch, self.max_n_nodes
                    )
                    n_nodes = node_mask.sum(-1)
                else:
                    loader = (
                        self.save_init_dl_val
                        if fb == "f"
                        else self.save_final_dl_val
                    )
                    batch = next(loader)
                    batch, node_mask = utils.data_to_dense(
                        batch, self.max_n_nodes
                    )
                    n_nodes = node_mask.sum(-1)
                if self.cfg.virtual_node:
                    batch.X = torch.cat(
                        [~node_mask.unsqueeze(-1), batch.X],
                        dim=-1,
                    )
                    node_mask = (
                        torch.ones_like(node_mask).to(batch.X.device).bool()
                    )
                    n_nodes = node_mask.sum(-1)
            else:
                batch_size = self.cfg.batch_size
                n_nodes = self.nodes_dist.sample_n(batch_size, self.device)
                arange = (
                    torch.arange(self.max_n_nodes, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                node_mask = arange < n_nodes.unsqueeze(1)
                if self.cfg.virtual_node:
                    node_mask = (
                        torch.ones_like(node_mask).to(self.device).bool()
                    )
                    n_nodes = node_mask.sum(-1)

                batch = utils.PlaceHolder(
                    X=self.limit_dist.X.repeat(
                        batch_size, self.max_n_nodes, 1
                    ).to(self.device),
                    E=self.limit_dist.E.repeat(
                        batch_size,
                        self.max_n_nodes,
                        self.max_n_nodes,
                        1,
                    ).to(self.device),
                    y=None,
                    charge=None,
                    n_nodes=n_nodes,
                )
                batch = batch.sample(onehot=True, node_mask=node_mask)

            batch.mask(node_mask=node_mask)
            x_tot, _, _, _ = self.langevin.record_langevin_seq(
                sample_net,
                batch,
                node_mask=node_mask,
                ipf_it=n,
                sample=True,
            )

            (
                X_accumulated_abs,
                E_accumulated_abs,
                X_accumulated_ratio,
                E_accumulated_ratio,
            ) = accumulated_abs_trajectory(x_tot, n_nodes, virtual=False)
            X_nbr_changes += X_accumulated_abs
            E_nbr_changes += E_accumulated_abs
            X_ratio_changes += X_accumulated_ratio
            E_ratio_changes += E_accumulated_ratio

            init_samples.append(x_tot.get_data(0, dim=1).collapse())
            samples.append(x_tot.get_data(-1, dim=1).collapse())
            batches.append(batch)

            if self.cfg.virtual_node:
                node_mask = samples[-1].X > 0
                n_nodes = node_mask.sum(-1)

            all_n_nodes.append(n_nodes)

            chains = utils.PlaceHolder(
                X=x_tot.X[:chains_to_save],
                E=x_tot.E[:chains_to_save],
                charge=None,
                y=None,
            )

            samples_to_generate -= len(n_nodes)
            i += 1
        # merge things together
        samples = utils.PlaceHolder(
            X=torch.cat([s.X for s in samples], dim=0)[:samples_to_return],
            E=torch.cat([s.E for s in samples], dim=0)[:samples_to_return],
            charge=None,
            y=None,
        )
        init_samples = utils.PlaceHolder(
            X=torch.cat([s.X for s in init_samples], dim=0)[:samples_to_return],
            E=torch.cat([s.E for s in init_samples], dim=0)[:samples_to_return],
            charge=None,
            y=None,
        )
        batches = utils.PlaceHolder(
            X=torch.cat([b.X for b in batches], dim=0)[:samples_to_return],
            E=torch.cat([b.E for b in batches], dim=0)[:samples_to_return],
            charge=None,
            y=None,
        )
        all_n_nodes = torch.cat(all_n_nodes, dim=0)[:samples_to_return]

        # return batch, samples, chains, all_n_nodes, all_node_masks
        return (
            batch,
            samples,
            chains,
            all_n_nodes,
            init_samples,
            (
                X_nbr_changes,
                E_nbr_changes,
                X_ratio_changes / i,
                E_ratio_changes / i,
            ),
        )

    def save_step(self, i, n, fb):
        """
        Step for sampling and for saving
        """
        if not self.cfg.test:
            samples_to_save = self.cfg.samples_to_save
            chains_to_save = self.cfg.chains_to_save
        else:
            samples_to_save = self.cfg.final_samples_to_save
            chains_to_save = self.cfg.final_chains_to_save

        if self.accelerator.is_local_main_process:
            # if not self.cfg.test:
            if self.cfg.ema:
                sample_net = self.ema_helpers[fb].ema_copy(
                    self.net[fb]
                )  # TODO: this may pose a problem for test
            else:
                sample_net = self.net[fb]

            name_net = "net" + "_" + fb + "_" + str(n) + "_" + str(i) + ".ckpt"
            name_net_ckpt = "./checkpoints/" + name_net

            if self.cfg.dataparallel:
                torch.save(
                    self.net[fb].module.state_dict(),
                    name_net_ckpt,
                )
            else:
                torch.save(self.net[fb].state_dict(), name_net_ckpt)

            if self.cfg.ema:
                name_net = (
                    "sample_net"
                    + "_"
                    + fb
                    + "_"
                    + str(n)
                    + "_"
                    + str(i)
                    + ".ckpt"
                )
                name_net_ckpt = "./checkpoints/" + name_net
                if self.cfg.dataparallel:
                    torch.save(
                        sample_net.module.state_dict(),
                        name_net_ckpt,
                    )
                else:
                    torch.save(sample_net.state_dict(), name_net_ckpt)

            # generation
            self.set_seed(seed=0 + self.accelerator.process_index)
            # batch, samples, chains, n_nodes, node_masks = self.generate_graphs(
            (
                batch,
                samples,
                chains,
                n_nodes,
                init_samples,
                acc_metrics,
            ) = self.generate_graphs(fb, sample_net, n, test=self.cfg.test)

            to_plot = utils.PlaceHolder(
                X=samples.X,
                E=samples.E,
                charge=None,
                y=None,
                n_nodes=n_nodes,
            )
            if self.cfg.virtual_node:
                to_plot.X -= 1
                print("virtual node", to_plot.X.min())

            X = to_plot.X
            E = to_plot.E

            generated_list = []
            for l in range(X.shape[0]):
                if self.cfg.virtual_node:
                    cur_mask = X[l] >= 0
                else:
                    cur_mask = (
                        torch.arange(X.size(-1), device=self.device)
                        < n_nodes[l]
                    )

                atom_types = X[l, cur_mask].cpu()
                edge_types = E[l, cur_mask][:, cur_mask].cpu()
                generated_list.append([atom_types, edge_types])

            # if self.cfg.transfer and self.cfg.test:
            if self.cfg.transfer:
                # if self.cfg.transfer:
                if self.cfg.virtual_node:
                    init_samples.X -= 1
                    print(
                        "virtual node for source graphs",
                        init_samples.X.min(),
                    )
                init_list = []
                for l in range(init_samples.X.shape[0]):
                    if self.cfg.virtual_node:
                        cur_mask = init_samples.X[l] >= 0
                    else:
                        cur_mask = (
                            torch.arange(
                                init_samples.X.size(-1),
                                device=self.device,
                            )
                            < n_nodes[l]
                        )

                    atom_types = init_samples.X[l, cur_mask].cpu()
                    edge_types = init_samples.E[l, cur_mask][:, cur_mask].cpu()
                    init_list.append([atom_types, edge_types])
            else:
                init_list = None

            print("visualizing graphs...")
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"im/step{str(i)}_iter{str(n)}_{fb}/",
            )

            self.visualization_tools.visualize(
                result_path,
                graph_list=generated_list[:samples_to_save],
                num_graphs_to_visualize=min(
                    len(generated_list), samples_to_save
                ),
                fb=fb,
            )

            print("Visualizing chains...")

            result_path = os.path.join(
                current_path,
                f"predict_chains_{fb}/ipf{n}_step{i}",
            )

            chains.X = torch.concatenate(
                (
                    batch.X.unsqueeze(1)[:chains_to_save],
                    chains.X,
                ),
                dim=1,
            )
            chains.E = torch.concatenate(
                (
                    batch.E.unsqueeze(1)[:chains_to_save],
                    chains.E,
                ),
                dim=1,
            )
            chains_to_save = min(chains_to_save, chains.X.shape[1])
            _ = self.visualization_tools.visualize_chains(
                result_path,
                chains=chains,
                num_nodes=n_nodes,
                local_rank=0,
                num_chains_to_visualize=chains_to_save,
                fb=fb,
                transfer=self.cfg.transfer,
                virtual_node=self.cfg.virtual_node,
            )

            # in this case, if fb=='f' and not transfer, then the metrics will become None
            # import pdb; pdb.set_trace()
            test_sampling_metrics = (
                self.test_sampling_metrics
                if fb == "b"
                else self.tf_test_sampling_metrics
            )
            val_sampling_metrics = (
                self.val_sampling_metrics
                if fb == "b"
                else self.tf_val_sampling_metrics
            )

            # using the absolute trajectory for discrete data with one-hot encoding is more intuitive
            X_abs, E_abs, node_mask, edge_mask = abs_trajectory(
                init_samples,
                samples,
                n_nodes,
                virtual=self.cfg.virtual_node,
            )
            X_acc_abs, E_acc_abs, X_acc_ratio, E_acc_ratio = acc_metrics

            if test_sampling_metrics is not None:
                test_to_log = test_sampling_metrics.compute_all_metrics(
                    generated_list,
                    current_epoch=0,
                    local_rank=0,
                    fb=fb,
                    i=np.round(i / (self.num_iter + 1), 2),
                    source_graphs=init_list,
                )

                test_to_log[f"{fb}_X_change"] = (
                    X_abs.item()
                )  # the nbr of nodes changed
                test_to_log[f"{fb}_E_change"] = (
                    E_abs.item()
                )  # the nbr of edges changed
                test_to_log[f"{fb}_graph_change"] = (
                    X_abs + E_abs
                ).item()  # the nbr of nodes changed + the nbr of edges changed
                test_to_log[f"{fb}_X_change_ratio"] = (
                    X_abs / node_mask.sum()
                ).item()  # the ratio of nodes changed
                test_to_log[f"{fb}_E_change_ratio"] = (
                    E_abs / edge_mask.sum()
                ).item()  # the ratio of edges changed
                test_to_log[f"{fb}_change_ratio"] = (
                    (X_abs + E_abs) / (node_mask.sum() + edge_mask.sum())
                ).item()  # the ratio of nodes changed + the ratio of edges changed
                test_to_log[f"{fb}_X_acc_change"] = (
                    X_acc_abs.item()
                )  # the nbr of nodes changed along the trajectory
                test_to_log[f"{fb}_E_acc_change"] = (
                    E_acc_abs.item()
                )  # the nbr of edges changed along the trajectory
                test_to_log[f"{fb}_X_acc_change_ratio"] = (
                    X_acc_ratio.item()
                )  # the ratio of nodes changed along the trajectory
                test_to_log[f"{fb}_E_acc_change_ratio"] = (
                    E_acc_ratio.item()
                )  # the ratio of edges changed along the trajectory
                print(test_to_log)

                # save results for testing
                print("saving results for testing")
                current_path = os.getcwd()
                res_path = os.path.join(
                    current_path,
                    f"test_{fb}_{n}.json",
                )

                with open(res_path, "w") as file:
                    json.dump(test_to_log, file)

                res_path = os.path.join(
                    current_path,
                    f"val_{fb}_{n}.txt",
                )

                for k in test_to_log:
                    with open(res_path, "a") as file:
                        file.write(f"{k}: {test_to_log[k]}\n")

                if wandb.run:
                    wandb.log(test_to_log, commit=False)

            elif val_sampling_metrics is not None:
                val_to_log = val_sampling_metrics.compute_all_metrics(
                    generated_list,
                    current_epoch=0,
                    local_rank=0,
                    fb=fb,
                    i=np.round(i / (self.num_iter + 1), 2),
                    source_graphs=init_list,
                )

                val_to_log[f"{fb}_X_change"] = (
                    X_abs.item()
                )  # the nbr of nodes changed
                val_to_log[f"{fb}_E_change"] = (
                    E_abs.item()
                )  # the nbr of edges changed
                val_to_log[f"{fb}_graph_change"] = (
                    X_abs + E_abs
                ).item()  # the nbr of nodes changed + the nbr of edges changed
                val_to_log[f"{fb}_X_change_ratio"] = (
                    X_abs / node_mask.sum()
                ).item()  # the ratio of nodes changed
                val_to_log[f"{fb}_E_change_ratio"] = (
                    E_abs / edge_mask.sum()
                ).item()  # the ratio of edges changed
                val_to_log[f"{fb}_change_ratio"] = (
                    (X_abs + E_abs) / (node_mask.sum() + edge_mask.sum())
                ).item()  # the ratio of nodes changed + the ratio of edges changed
                val_to_log[f"{fb}_X_acc_change"] = (
                    X_acc_abs.item()
                )  # the nbr of nodes changed along the trajectory
                val_to_log[f"{fb}_E_acc_change"] = (
                    E_acc_abs.item()
                )  # the nbr of edges changed along the trajectory
                val_to_log[f"{fb}_X_acc_change_ratio"] = (
                    X_acc_ratio.item()
                )  # the ratio of nodes changed along the trajectory
                val_to_log[f"{fb}_E_acc_change_ratio"] = (
                    E_acc_ratio.item()
                )  # the ratio of edges changed along the trajectory
                print(val_to_log)

                # save results for validating
                print("saving results for validating")
                current_path = os.getcwd()
                res_path = os.path.join(
                    current_path,
                    f"val_{fb}_{n}.json",
                )

                with open(res_path, "w") as file:
                    json.dump(val_to_log, file)

                res_path = os.path.join(
                    current_path,
                    f"val_{fb}_{n}.txt",
                )

                for k in val_to_log:
                    with open(res_path, "a") as file:
                        file.write(f"{k}: {val_to_log[k]}\n")

                if wandb.run:
                    wandb.log(val_to_log, commit=False)

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
        new_dl = self.new_cacheloader(forward_or_backward, n, self.cfg.ema)

        if not self.cfg.use_prev_net:
            self.build_models(forward_or_backward)
            self.update_ema(forward_or_backward)

        self.build_optimizers()
        self.accelerate(forward_or_backward)

        for i in tqdm(range(self.num_iter)):
            """
            training step
            """
            self.set_seed(seed=n * self.num_iter + i)

            x, out, clean, gammas_expanded, times_expanded = next(new_dl)

            if self.cfg.virtual_node:
                # we do not consider node mask when using virtual nodes
                n_nodes_x = (
                    torch.ones_like(x[4]).to(self.device) * self.max_n_nodes
                )
                n_nodes_out = (
                    torch.ones_like(out[4]).to(self.device) * self.max_n_nodes
                )
                n_nodes_clean = (
                    torch.ones_like(clean[4]).to(self.device) * self.max_n_nodes
                )
            else:
                n_nodes_x = x[4]
                n_nodes_out = out[4]
                n_nodes_clean = clean[4]

            x = PlaceHolder(
                X=x[0],
                E=x[1],
                y=x[2],
                charge=x[3],
                n_nodes=n_nodes_x,
            )
            out = PlaceHolder(
                X=out[0],
                E=out[1],
                y=out[2],
                charge=out[3],
                n_nodes=n_nodes_out,
            )
            clean = PlaceHolder(
                X=clean[0],
                E=clean[1],
                y=clean[2],
                charge=clean[3],
                n_nodes=n_nodes_clean,
            )

            eval_steps = self.T - times_expanded
            pred = self.forward_graph(
                self.net[forward_or_backward], x, eval_steps
            )

            pred_R = pred.copy()

            # dp = r * dt
            pred_clean = pred.copy()
            pred_clean.X = pred_clean.X * times_expanded[:, None, :]
            pred_clean.E = pred_clean.E * times_expanded[:, None, None, :]
            # change the value for the diagonal
            pred_clean.X.scatter_(-1, x.X.argmax(-1)[:, :, None], 0.0)
            pred_clean.E.scatter_(-1, x.E.argmax(-1)[:, :, :, None], 0.0)
            pred_clean.X.scatter_(
                -1,
                x.X.argmax(-1)[:, :, None],
                (1.0 - pred_clean.X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
            )
            pred.E.scatter_(
                -1,
                x.E.argmax(-1)[:, :, :, None],
                (1.0 - pred_clean.E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
            )
            # normalization
            pred_clean.X = pred_clean.X + 1e-6
            pred_clean.E = pred_clean.E + 1e-6
            pred_clean.X = pred_clean.X / pred_clean.X.sum(-1, keepdim=True)
            pred_clean.E = pred_clean.E / pred_clean.E.sum(-1, keepdim=True)

            # dp = r * dt
            pred.X = pred.X * gammas_expanded[:, None, :]
            pred.E = pred.E * gammas_expanded[:, None, None, :]
            # change the value for the diagonal
            pred.X.scatter_(-1, x.X.argmax(-1)[:, :, None], 0.0)
            pred.E.scatter_(-1, x.E.argmax(-1)[:, :, :, None], 0.0)
            pred.X.scatter_(
                -1,
                x.X.argmax(-1)[:, :, None],
                (1.0 - pred.X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
            )
            pred.E.scatter_(
                -1,
                x.E.argmax(-1)[:, :, :, None],
                (1.0 - pred.E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
            )

            # normalization
            pred.X = pred.X + 1e-6
            pred.E = pred.E + 1e-6
            pred.X = pred.X / pred.X.sum(-1, keepdim=True)
            pred.E = pred.E / pred.E.sum(-1, keepdim=True)

            # # node and edge losses are not specifically used here
            _, _, loss = self.compute_loss(
                pred,
                out,
                pred_clean,
                clean,
                times_expanded,
                pred_R,
            )

            num_log = 5000
            if self.num_steps <= num_log:
                gap = 1
            else:
                gap = self.num_steps // num_log

            if wandb.run and (i % gap == 0):
                wandb.log(
                    {
                        f"train/loss_{forward_or_backward}": loss.detach()
                        .cpu()
                        .numpy()
                        .item()
                    },
                )

            self.accelerator.backward(loss)
            if self.cfg.grad_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.net[forward_or_backward].parameters(),
                    self.grad_clip_value,
                )

            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad()
            if self.cfg.ema:
                self.ema_helpers[forward_or_backward].update(
                    self.net[forward_or_backward]
                )

            if (i % self.cfg.cache_refresh_stride == 0) and (i > 0):
                new_dl = None
                torch.cuda.empty_cache()
                new_dl = self.new_cacheloader(
                    forward_or_backward, n, self.cfg.ema
                )

        if n % self.cfg.save_every_ipf == 0:
            self.save_step(i, n, forward_or_backward)

        new_dl = None
        self.clear()

    def compute_loss(self, pred, out, pred_clean, clean, t, pred_R):
        use_edge_loss = True
        clean_node_count = clean.node_mask.sum(-1)
        node_count = out.node_mask.sum(-1)

        if not use_edge_loss and self.cfg.virtual_node:
            node_count = (out.X.argmax(-1) > 0).sum(-1)
            clean_node_count = (clean.X.argmax(-1) > 0).sum(-1)

        clean_edge_count = (clean_node_count - 1) * clean_node_count
        edge_count = (node_count - 1) * node_count

        # Calculate the losses
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        pred.X = torch.log(pred.X + 1e-6)
        pred.E = torch.log(pred.E + 1e-6)
        node_loss = self.cfg.model.lambda_train[0] * ce_loss(
            pred.X.permute((0, 2, 1)), out.X.permute((0, 2, 1))
        )
        edge_loss = self.cfg.model.lambda_train[1] * ce_loss(
            pred.E.permute((0, 3, 1, 2)),
            out.E.permute((0, 3, 1, 2)),
        )

        loss = PlaceHolder(
            X=node_loss.unsqueeze(-1),
            E=edge_loss.unsqueeze(-1),
            y=None,
        )
        if not self.cfg.virtual_node:
            loss = loss.mask(out.node_mask)
        else:
            if not use_edge_loss:
                loss = loss.mask(
                    mask_node=not self.cfg.virtual_node,
                    node_mask=(out.X.argmax(-1) > 0),
                )
        loss = (loss.X / node_count).sum() + (
            loss.E / edge_count
        ).sum() * self.cfg.edge_weight

        # sparsity loss - V2
        weight = self.cfg.reg_weight
        loss = (
            loss
            + torch.norm(pred_R.X, 1.0) * weight / node_count.sum()
            + torch.norm(pred_R.E, 1.0)
            * weight
            / edge_count.sum()
            * self.cfg.edge_weight
        )

        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        pred_clean.X = torch.log(pred_clean.X + 1e-6)
        pred_clean.E = torch.log(pred_clean.E + 1e-6)
        clean_node_loss = self.cfg.model.lambda_train[0] * ce_loss(
            pred_clean.X.permute((0, 2, 1)),
            clean.X.permute((0, 2, 1)),
        )
        clean_edge_loss = self.cfg.model.lambda_train[1] * ce_loss(
            pred_clean.E.permute((0, 3, 1, 2)),
            clean.E.permute((0, 3, 1, 2)),
        )
        # clean_node_loss = clean_node_loss / t[:]
        # clean_edge_loss = clean_edge_loss / t[:, None]
        clean_node_loss = clean_node_loss / torch.exp(t[:])
        clean_edge_loss = clean_edge_loss / torch.exp(t[:, None])

        clean_loss = PlaceHolder(
            X=clean_node_loss.unsqueeze(-1),
            E=clean_edge_loss.unsqueeze(-1),
            y=None,
        )
        if not self.cfg.virtual_node:
            clean_loss = clean_loss.mask(clean.node_mask)
        else:
            if not use_edge_loss:
                clean_loss = clean_loss.mask(
                    mask_node=not self.cfg.virtual_node,
                    node_mask=(clean.X.argmax(-1) > 0),
                )
        clean_loss = (clean_loss.X / clean_node_count).sum() + (
            clean_loss.E / clean_edge_count
        ).sum() * self.cfg.edge_weight

        loss = loss + self.cfg.clean_loss_weight * clean_loss

        if pred.charge.numel() > 0:
            loss = loss + self.cfg.model.lambda_train[0] * F.mse_loss(
                pred.E, out.E
            )

        return node_loss, edge_loss, loss

    def forward_graph(self, net, z_t, t):
        # step 1: calculate extra features
        assert z_t.node_mask is not None

        model_input = z_t.copy()
        with torch.no_grad():
            if self.cfg.virtual_node:
                X = z_t.X.clone()
                z_t.X = z_t.X[..., 1:]
            extra_features = self.extra_features(z_t)
            extra_domain_features = self.domain_features(z_t)
            if self.cfg.virtual_node:
                z_t.X = X

        model_input.X = torch.cat(
            (z_t.X, extra_features.X, extra_domain_features.X),
            dim=2,
        ).float()
        model_input.E = torch.cat(
            (z_t.E, extra_features.E, extra_domain_features.E),
            dim=3,
        ).float()
        model_input.y = torch.hstack(
            (
                z_t.y,
                extra_features.y,
                extra_domain_features.y,
                t,
            )
        ).float()

        res = net(model_input)

        return res

    def train(self):
        print("Training...")

        for n in range(self.checkpoint_it, self.n_ipf + 1):
            print("IPF iteration: " + str(n) + "/" + str(self.n_ipf))
            self.ipf_step("b", n)
            self.ipf_step("f", n)

    def test(self):
        print("Testing...")

        if len(self.cfg.sample_checkpoint_f) > 1:
            self.save_step(0, 0, "f")
        if len(self.cfg.sample_checkpoint_b) > 1:
            self.save_step(0, 0, "b")
