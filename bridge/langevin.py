import os
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import utils


def get_noise(limit_dist, x_k, node_mask):
    batch_size = x_k.X.shape[0]
    n_nodes = x_k.n_nodes
    max_n_nodes = x_k.X.shape[1]
    device = x_k.X.device

    limit_X = limit_dist.X
    limit_E = limit_dist.E

    batch = utils.PlaceHolder(
        X=limit_X.repeat(batch_size, max_n_nodes, 1).to(device),
        E=limit_E.repeat(batch_size, max_n_nodes, max_n_nodes, 1).to(device),
        y=None,
        charge=None,
        n_nodes=n_nodes,
    )

    return batch.mask(node_mask)


def ornstein_ulhenbeck(x, gradx, gamma, graph=False):
    z = torch.randn(x.shape, device=x.device)
    if graph:
        upper_triangle = torch.triu(z, diagonal=1)
        z = upper_triangle + upper_triangle.transpose(0, 1)
        z = symmetrize_graphs(z)
    xout = x + gamma * gradx + torch.sqrt(2 * gamma) * z
    return xout


def symmetrize_graphs(tensor):
    batch_size, channels, height, width = tensor.shape
    mask = torch.triu(torch.ones(height, width, dtype=torch.bool), diagonal=1)
    mask = mask[None, None, :, :]  # Add dimensions for batch and channel
    mask = mask.expand(batch_size, channels, height, width).to(tensor.device)

    tensor = tensor * mask
    tensor = tensor + tensor.permute(0, 1, 3, 2)

    return tensor


class Langevin(torch.nn.Module):
    def __init__(
        self,
        num_steps,
        max_n_nodes,
        gammas,
        time_sampler,
        limit_dist=None,
        device=None,
        graph=False,
        extra_features=None,
        domain_features=None,
        tf_extra_features=None,
        tf_domain_features=None,
        virtual_node=False,
        noise_level=1.0,
    ):
        super().__init__()
        # self.std_final = utils.PlaceHolder(
        #     X=torch.sqrt(self.var_final.X), E=torch.sqrt(self.var_final.E), y=None
        # )
        self.graph = graph
        self.limit_dist = limit_dist
        self.virtual_node = virtual_node
        self.noise_level = noise_level

        # extra fetaures
        self.extra_features = extra_features
        self.domain_features = domain_features
        self.tf_extra_features = tf_domain_features
        self.tf_domain_features = tf_domain_features

        self.num_steps = num_steps  # num diffusion steps
        self.max_n_nodes = max_n_nodes  # max_n_nodes of object to diffuse
        self.gammas = gammas.float()  # schedule

        if device is not None:
            self.device = device
        else:
            self.device = gammas.device

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas, 0).to(self.device).float()
        self.time_sampler = time_sampler

    def record_init_langevin(self, x, node_mask):
        bs = x.X.shape[0]
        dx = x.X.shape[-1]  # for virtual nodes, there is an extra dimension
        de = x.E.shape[-1]

        x_tot = utils.PlaceHolder(
            X=torch.Tensor(bs, self.num_steps, self.max_n_nodes, dx).to(self.device),
            E=torch.Tensor(
                bs, self.num_steps, self.max_n_nodes, self.max_n_nodes, de
            ).to(self.device),
            y=None,
        )

        out = utils.PlaceHolder(
            X=torch.Tensor(bs, self.num_steps, self.max_n_nodes, dx).to(self.device),
            E=torch.Tensor(
                bs, self.num_steps, self.max_n_nodes, self.max_n_nodes, de
            ).to(self.device),
            y=None,
        )

        times_expanded = self.time.reshape((1, self.num_steps, 1)).repeat((bs, 1, 1))
        gammas_expanded = self.gammas.reshape((1, self.num_steps, 1)).repeat((bs, 1, 1))

        x_k = x.copy()
        noise = get_noise(self.limit_dist, x_k, x.node_mask)
        for k in range(self.num_steps):
            out = out.place(x_k, k)
            gamma = self.gammas[k]
            # The line `x_k = x_k.scale(1 - gamma/10).add(noise.scale(gamma/10))`
            # in the code snippet is performing a scaling and adding operation on
            # the variable `x_k`.
            # x_k = x_k.scale(1 - gamma/10).add(noise.scale(gamma/10))
            # x_k = x_k.scale(1 - gamma/2).add(noise.scale(gamma/2))
            # x_k = x_k.scale(1 - gamma).add(noise.scale(gamma))
            # The line `# x_k = x_k.scale(1 - gamma*3).add(noise.scale(gamma*3))`
            # is currently commented out in the code snippet. If you were to
            # uncomment it, this line would perform a scaling and adding operation
            # on the variable `x_k`.
            # x_k = x_k.scale(1 - gamma/2).add(noise.scale(gamma/2))
            # x_k = x_k.scale(1 - gamma).add(noise.scale(gamma))
            x_k = x_k.scale(1 - gamma * self.noise_level).add(noise.scale(gamma * self.noise_level))
            # if k < self.num_steps/2:
            #     x_k = x_k.scale(1 - gamma*2).add(noise.scale(gamma*2))
            # else:
            #     x_k = noise
            # x_k = x_k.scale(1 - gamma / 3).add(noise.scale(gamma / 3))
            # The line `# x_k = x_k.scale(1 - gamma*2).add(noise.scale(gamma*2))`
            # is currently commented out in the code snippet. If you were to
            # uncomment it, this line would perform a scaling and adding operation
            # on the variable `x_k`.
            x_k.X = x_k.X.clamp(0.0, 1.0)
            x_k.E = x_k.E.clamp(0.0, 1.0)
            if self.virtual_node:
                x_k = x_k.sample(onehot=True, node_mask=torch.ones(x_k.X.shape[:-1]).to(x_k.X.device).bool())
            else:
                x_k = x_k.sample(onehot=True, node_mask=node_mask)

            x_tot = x_tot.place(x_k, k)

        return x_tot, out, gammas_expanded, times_expanded

    def record_langevin_seq(
        self, net, init_samples, node_mask, t_batch=None, ipf_it=0, sample=False, time=None, gammas=None
    ):
        bs = init_samples.X.shape[0]
        dx = init_samples.X.shape[-1]  # for virtual nodes, there is an extra dimension
        de = init_samples.E.shape[-1]

        x_tot = utils.PlaceHolder(
            X=torch.Tensor(bs, self.num_steps, self.max_n_nodes, dx).to(self.device),
            E=torch.Tensor(
                bs, self.num_steps, self.max_n_nodes, self.max_n_nodes, de
            ).to(self.device),
            y=None,
            node_mask=node_mask,
        )

        out = utils.PlaceHolder(
            X=torch.Tensor(bs, self.num_steps, self.max_n_nodes, dx).to(self.device),
            E=torch.Tensor(
                bs, self.num_steps, self.max_n_nodes, self.max_n_nodes, de
            ).to(self.device),
            y=None,
            node_mask=node_mask,
        )

        if time is None:
            time = self.time
        if gammas is None:
            gammas = self.gammas
        
        times_expanded = time.reshape((1, self.num_steps, 1)).repeat((bs, 1, 1))
        gammas_expanded = gammas.reshape((1, self.num_steps, 1)).repeat((bs, 1, 1))

        x = init_samples.copy()
        # for k in range(self.num_steps):
        num_repeat = 1
        for i in range(self.num_steps*num_repeat):

            out_save = i % num_repeat == 0
            k = i // num_repeat

            if out_save:
                out = out.place(x, k)

            # t = times_expanded[:, k, :]
            # gamma = gammas_expanded[:, k, :]
            t = torch.ones_like(times_expanded[:, k, :], device=self.device) * i / self.num_steps / num_repeat
            gamma = gammas_expanded[:, k, :] / num_repeat
            # / k * i / self.num_steps*num_repeat
            with torch.no_grad():
                pred = self.forward_graph(net, x, t)

            pred.X = pred.X * gamma[:, None, :]
            pred.E = pred.E * gamma[:, None, None, :]

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
            # The normalization should be automatic here
            # Added to be consistent the the training process
            pred.X = (pred.X / pred.X.sum(-1, keepdim=True)).float()
            pred.E = (pred.E / pred.E.sum(-1, keepdim=True)).float()

            if self.virtual_node:
                x = pred.sample(onehot=True, node_mask=torch.ones(x.X.shape[:-1]).to(x.X.device).bool())
            else:
                x = pred.sample(onehot=True, node_mask=node_mask)

            # if out_save and k > 0:
            if i % num_repeat == num_repeat - 1:
                x_tot = x_tot.place(x, k)

        return x_tot, out, gammas_expanded, times_expanded

    def forward_graph(self, net, z_t, t):
        # step 1: calculate extra features
        assert z_t.node_mask is not None

        model_input = z_t.copy()
        with torch.no_grad():
            if self.virtual_node:
                X = z_t.X.clone()
                z_t.X = z_t.X[..., 1:]
            extra_features = self.extra_features(z_t)
            extra_domain_features = self.domain_features(z_t)
            if self.virtual_node:
                z_t.X = X

        model_input.X = torch.cat(
            (z_t.X, extra_features.X, extra_domain_features.X), dim=2
        ).float()
        model_input.E = torch.cat(
            (z_t.E, extra_features.E, extra_domain_features.E), dim=3
        ).float()
        model_input.y = torch.hstack(
            (z_t.y, extra_features.y, extra_domain_features.y, t)
        ).float()

        res = net(model_input)
        return res
