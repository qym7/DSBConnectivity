import os
import copy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import utils


def grad_gauss(x, m, var):
    xout = (x - m) / var
    return -xout

def graph_grad_gauss(graph_data, m, var):
    grad = utils.PlaceHolder(
        X=-(graph_data.X - m.X) / var.X,
        E=-(graph_data.E - m.E) / var.E,
        y=None,
        charge=None,
        node_mask=graph_data.node_mask
    )
    # grad = utils.PlaceHolder(
    #     X=-(graph_data.X),
    #     E=-(graph_data.E),
    #     y=None,
    #     charge=None,
    #     node_mask=graph_data.node_mask
    # )
    grad = grad.mask()

    return grad


def calculate_update(x, gamma, mean_final, var_final, node_mask):
    # calculation
    grad = graph_grad_gauss(x, mean_final, var_final)
    t_old = x.add(grad.scale(gamma))
    z = x.randn_like()
    x = t_old.add(z.scale(torch.sqrt(2 * gamma)))
    grad = graph_grad_gauss(x, mean_final, var_final)
    t_new = x.add(grad.scale(gamma))
    out = t_old.minus(t_new)

    # mask
    x = x.mask(node_mask)
    out = out.mask(node_mask)

    return x, out


def ornstein_ulhenbeck(x, gradx, gamma, graph=False):
    z = torch.randn(x.shape, device=x.device)
    if graph:
        upper_triangle = torch.triu(z, diagonal=1)
        z = upper_triangle + upper_triangle.transpose(0, 1)
        z = symmetrize_graphs(z)
    xout = x + gamma * gradx + \
        torch.sqrt(2 * gamma) * z
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

    def __init__(self, num_steps, max_n_nodes, gammas, time_sampler, device=None,
                 mean_final=torch.tensor([0., 0.]), var_final=torch.tensor([.5, .5]), mean_match=True,
                 graph=False, extra_features=None, domain_features=None,
                 tf_extra_features=None, tf_domain_features=None, thres=1.0):
        super().__init__()

        self.mean_match = mean_match
        self.mean_final = mean_final
        self.var_final = var_final
        self.thres = thres
        self.std_final = utils.PlaceHolder(
            X=torch.sqrt(self.var_final.X),
            E=torch.sqrt(self.var_final.E),
            y=None
        )
        self.graph = graph
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
        mean_final = self.mean_final
        var_final = self.var_final
        std_final = self.std_final

        x_k = x.copy()
        # x_k.E = x_k.E[:,:,:,-1].unsqueeze(-1)

        bs = x_k.X.shape[0]
        dx = x_k.X.shape[-1]
        de = x_k.E.shape[-1]

        x_tot = utils.PlaceHolder(
            X=torch.Tensor(bs, self.num_steps, self.max_n_nodes, dx).to(self.device),
            E=torch.Tensor(bs, self.num_steps, self.max_n_nodes, self.max_n_nodes, de).to(self.device),
            y=None
        )
        out = utils.PlaceHolder(
            X=torch.Tensor(bs, self.num_steps, self.max_n_nodes, dx).to(self.device),
            E=torch.Tensor(bs, self.num_steps, self.max_n_nodes, self.max_n_nodes, de).to(self.device),
            y=None
        )
        # import pdb; pdb.set_trace()
        steps_expanded = self.time.reshape((1, self.num_steps, 1)).repeat((bs, 1, 1))  # TODO: this might not be correct

        for k in range(self.num_steps):
            gamma = self.gammas[k]
            x_k, out_k = calculate_update(x_k, gamma, mean_final, var_final, node_mask)
            x_tot.place(x_k, k)
            out.place(out_k, k)
            # print(k, x_k.E[0,0,1], out_k.E[0,0,1])

        return x_tot, out, steps_expanded

    def record_langevin_seq(self, net, init_samples, node_mask, fb='f', ipf_it=0, sample=False):
        bs = init_samples.X.shape[0]
        dx = init_samples.X.shape[-1]
        de = init_samples.E.shape[-1]

        x_tot = utils.PlaceHolder(
            X=torch.Tensor(bs, self.num_steps, self.max_n_nodes, dx).to(self.device),
            E=torch.Tensor(bs, self.num_steps, self.max_n_nodes, self.max_n_nodes, de).to(self.device),
            y=None,
            node_mask=node_mask
        )
        out = utils.PlaceHolder(
            X=torch.Tensor(bs, self.num_steps, self.max_n_nodes, dx).to(self.device),
            E=torch.Tensor(bs, self.num_steps, self.max_n_nodes, self.max_n_nodes, de).to(self.device),
            y=None,
            node_mask=node_mask
        )

        steps_expanded = self.time.reshape((1, self.num_steps, 1)).repeat((bs, 1, 1))
        x = init_samples.copy()

        for k in tqdm(range(self.num_steps)):
            t = steps_expanded[:, k, :]  # (bs, 1)
            gamma = self.gammas[k]
            if self.mean_match:
                t_old = self.forward_graph(net, x, t)

                if sample & (k == self.num_steps-1):
                    x = t_old
                else:
                    z = t_old.randn_like().scale(torch.sqrt(2 * gamma))
                    x = t_old.add(z)
                try:
                    t_new = self.forward_graph(net, x, t)
                except:
                    import pdb; pdb.set_trace()
            else:
                t_old = x.add(self.forward_graph(net, x, t))

                if sample & (k == self.num_steps-1):
                    x = t_old
                else:
                    z = t_old.randn_like()
                    z = z.scale(torch.sqrt(2 * gamma))
                    x = t_old.add(z)
                t_new = x.add(self.forward_graph(net, x, t))

            x = x.mask()
            t_old = t_old.mask()
            t_new = t_new.mask()
            out_k = t_old.minus(t_new)

            # x.X = torch.zeros_like(x.X, device=x_tot.X.device)
            # out_k.X = torch.zeros_like(out_k.X, device=out.X.device)
            x_tot = x_tot.place(x.copy(), k)
            out = out.place(out_k.copy(), k)

        return x_tot, out, steps_expanded


    def forward_graph(self, net, z_t, t):
        # step 1: calculate extra features
        assert z_t.node_mask is not None
        # z_t = z_t.clip(3)

        model_input = z_t.copy()
        with torch.no_grad():
            z_t_discrete = z_t.onehot(thres=self.thres)
            extra_features, _, _ = self.extra_features(z_t_discrete)
            extra_domain_features = self.domain_features(z_t_discrete)

        model_input.X = torch.cat(
            (z_t.X, z_t_discrete.X, extra_features.X, extra_domain_features.X), dim=2
        ).float()
        model_input.E = torch.cat(
            (z_t.E, z_t_discrete.E, extra_features.E, extra_domain_features.E), dim=3
        ).float()
        model_input.y = torch.hstack(
            (z_t.y, z_t_discrete.y, extra_features.y, extra_domain_features.y, t)
        ).float()

        res = net(model_input)

        return res
