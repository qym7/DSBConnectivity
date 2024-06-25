import torch
import os
import os.path as osp
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pathlib
from hydra.utils import get_original_cwd
import pytorch_lightning as pl

from ..utils import *

from .logger import CSVLogger, NeptuneLogger, Logger
from ..models import *
from ..models.gnn.transformer_model import GraphTransformer
from ..data.two_dim import two_dim_ds
from ..data.stackedmnist import Stacked_MNIST
from ..data.emnist import EMNIST
from ..data.celeba import CelebA

# from ..data.comm20 import SpectreGraphDataset
# from ..data.synthetic_dataset import SyntheticGraphDataset
from .plotters import TwoDPlotter, ImPlotter

cmp = lambda x: transforms.Compose([*x])


def get_plotter(runner, args):
    dataset_tag = getattr(args, DATASET)
    if dataset_tag == DATASET_2D:
        return TwoDPlotter(num_steps=runner.num_steps, gammas=runner.langevin.gammas)
    else:
        return ImPlotter(plot_level=args.plot_level)


# Model
# --------------------------------------------------------------------------------


def get_graph_models(args, dataset_infos):
    # kwargs = {
    #     "n_layers": args.model.n_layers,
    #     "input_dims": args.model.input_dims,
    #     "hidden_dims": args.model.hidden_dims,
    #     "output_dims": args.model.output_dims,
    #     "sn_hidden_dim": args.model.sn_hidden_dim,
    #     "output_y": args.model.output_y,
    #     "dropout": args.model.dropout
    #     }
    kwargs = {
        "input_dims": dataset_infos.input_dims,
        "n_layers": args.model.n_layers,
        "hidden_mlp_dims": args.model.hidden_mlp_dims,
        "hidden_dims": args.model.hidden_dims,
        "output_dims": dataset_infos.output_dims,
        "dropout": args.model.dropout,
    }

    net_f, net_b = GraphTransformer(**kwargs), GraphTransformer(**kwargs)

    return net_f, net_b


# Dataset
# --------------------------------------------------------------------------------
def get_both_datamodules(cfg):
    pl.seed_everything(cfg.seed)
    # get datamodules for the initial and transfer dataset
    dataset_config = cfg["dataset"]
    train_metrics, domain_features, datamodule, datainfos = get_datamodules(
        dataset_config
    )
    if cfg.transfer:
        tf_dataset_config = cfg["dataset_transfer"]
        (
            tf_train_metrics,
            tf_domain_features,
            tf_datamodule,
            tf_datainfos,
        ) = get_datamodules(tf_dataset_config)
    else:
        tf_train_metrics, tf_domain_features, tf_datamodule, tf_datainfos = (
            None,
            None,
            None,
            None,
        )

    return (
        train_metrics,
        domain_features,
        datamodule,
        datainfos,
        tf_train_metrics,
        tf_domain_features,
        tf_datamodule,
        tf_datainfos,
    )


def get_datamodules(cfg):
    from ..metrics.abstract_metrics import TrainAbstractMetricsDiscrete
    from ..metrics.molecular_metrics import TrainMolecularMetricsDiscrete
    from ..diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
    from ..diffusion.extra_features_molecular import ExtraMolecularFeatures

    # step 1: get datamodules according to dataset name

    print("creating datasets")
    if cfg["name"] in ["sbm", "sbm_syn", "sbm_split", "comm20", "planar", "ego"]:
        from ..datasets.spectre_dataset_pyg import (
            SBMDataModule,
            SBMSynDataModule,
            Comm20DataModule,
            EgoDataModule,
            PlanarDataModule,
            SpectreDatasetInfos,
        )

        if cfg["name"] == "sbm":
            datamodule = SBMDataModule(cfg)
        if cfg["name"] == "sbm_syn":
            datamodule = SBMSynDataModule(cfg)
        elif cfg["name"] == "comm20":
            datamodule = Comm20DataModule(cfg)
        elif cfg["name"] == "ego":
            datamodule = EgoDataModule(cfg)
        else:
            datamodule = PlanarDataModule(cfg)

        dataset_infos = SpectreDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders

    elif cfg["name"] == "protein":
        from datasets import protein_dataset

        datamodule = protein_dataset.ProteinDataModule(cfg)
        dataset_infos = protein_dataset.ProteinInfos(datamodule=datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()

    elif cfg["name"] == "point_cloud":
        from datasets import point_cloud_dataset

        datamodule = point_cloud_dataset.PointCloudDataModule(cfg)
        dataset_infos = point_cloud_dataset.PointCloudInfos(datamodule=datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()

    elif cfg["name"] in ["qm9", "guacamol", "moses"]:
        if cfg["name"] == "qm9":
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9Infos(datamodule=datamodule, cfg=cfg)

        elif cfg["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.GuacamolInfos(datamodule, cfg)

        elif cfg.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MosesInfos(datamodule, cfg)
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.extra_features is not None:
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            domain_features = DummyExtraFeatures()

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["name"]))

    return train_metrics, domain_features, datamodule, dataset_infos


# Optimizer
# --------------------------------------------------------------------------------
def get_optimizers(net_f, net_b, lr):
    # return torch.optim.Adam(net_f.parameters(), lr=lr), torch.optim.Adam(net_b.parameters(), lr=lr)
    return (
        torch.optim.Adam(net_f.parameters(), lr=lr, amsgrad=True, weight_decay=1e-12),
        torch.optim.Adam(net_b.parameters(), lr=lr, amsgrad=True, weight_decay=1e-12),
    )


# Dataset
# --------------------------------------------------------------------------------

DATASET = "Dataset"
DATASET_TRANSFER = "Dataset_transfer"
DATASET_2D = "2d"
DATASET_CELEBA = "celeba"
DATASET_STACKEDMNIST = "stackedmnist"
DATASET_EMNIST = "emnist"
DATASET_COMM20 = "comm20"
DATASET_SBM1 = "sbm1"
DATASET_SBM2 = "sbm2"

# def get_datasets(args, split='train'):
#     dataset_tag = getattr(args, DATASET)
#     if args.transfer:
#         dataset_transfer_tag = getattr(args, DATASET_TRANSFER)
#     else:
#         dataset_transfer_tag = None

#     # INITIAL (DATA) DATASET
#     if dataset_tag == DATASET_COMM20:
#         base_path = pathlib.Path(get_original_cwd()).parents[0]
#         root_path = os.path.join(base_path, 'data/comm20/')
#         init_ds = SpectreGraphDataset(
#                     dataset_name='comm20',
#                     split=split,
#                     root=root_path,
#                     transform=None,
#                     pre_transform=None,
#                     pre_filter=None)

#     if dataset_tag == DATASET_SBM1:
#         base_path = pathlib.Path(get_original_cwd()).parents[0]
#         root_path = os.path.join(base_path, 'data/sym_sbm/')
#         init_ds = SyntheticGraphDataset(
#                     dataset_name='sbm',
#                     split=split,
#                     root=root_path,
#                     dataset_cfg=args.data,
#                     transform=None,
#                     pre_transform=None,
#                     pre_filter=None)

#     if dataset_transfer_tag == DATASET_SBM1:
#         base_path = pathlib.Path(get_original_cwd()).parents[0]
#         root_path = os.path.join(base_path, 'data/sym_sbm/')
#         final_ds = SyntheticGraphDataset(
#                     dataset_name='sbm',
#                     split=split,
#                     root=root_path,
#                     dataset_cfg=args.data_transfer,
#                     transform=None,
#                     pre_transform=None,
#                     pre_filter=None)
#         mean_final = torch.tensor(0.)
#         var_final = torch.tensor(1.*10**3)

#     if dataset_tag == DATASET_SBM2:
#         base_path = pathlib.Path(get_original_cwd()).parents[0]
#         root_path = os.path.join(base_path, 'data/sym_sbm/')
#         init_ds = SyntheticGraphDataset(
#                     dataset_name='sbm',
#                     split=split,
#                     root=root_path,
#                     dataset_cfg=args.data,
#                     transform=None,
#                     pre_transform=None,
#                     pre_filter=None)

#     if dataset_transfer_tag == DATASET_SBM2:
#         base_path = pathlib.Path(get_original_cwd()).parents[0]
#         root_path = os.path.join(base_path, 'data/sym_sbm/')
#         final_ds = SyntheticGraphDataset(
#                     dataset_name='sbm',
#                     split=split,
#                     root=root_path,
#                     dataset_cfg=args.data_transfer,
#                     transform=None,
#                     pre_transform=None,
#                     pre_filter=None)
#         mean_final = torch.tensor(0.)
#         var_final = torch.tensor(1.*10**3)

#     # 2D DATASET
#     if dataset_tag == DATASET_2D:
#         data_tag = args.data
#         npar = max(args.npar, args.cache_npar)
#         init_ds = two_dim_ds(npar, data_tag)

#     if dataset_transfer_tag == DATASET_2D:
#         data_tag = args.data_transfer
#         npar = max(args.npar, args.cache_npar)
#         final_ds = two_dim_ds(npar, data_tag)
#         mean_final = torch.tensor(0.)
#         var_final = torch.tensor(1.*10**3) #infty like

#     # CELEBA DATASET

#     if dataset_tag == DATASET_CELEBA:

#         train_transform = [transforms.CenterCrop(140), transforms.Resize(args.data.image_size), transforms.ToTensor()]
#         test_transform = [transforms.CenterCrop(140), transforms.Resize(args.data.image_size), transforms.ToTensor()]
#         if args.data.random_flip:
#             train_transform.insert(2, transforms.RandomHorizontalFlip())

#         root = os.path.join(args.data_dir, 'celeba')
#         init_ds = CelebA(root, split=split, transform=cmp(train_transform), download=False)

#     # MNIST DATASET
#     if dataset_tag ==  DATASET_STACKEDMNIST:
#         root = os.path.join(args.data_dir, 'mnist')
#         saved_file = os.path.join(root, "data.pt")
#         load = os.path.exists(saved_file)
#         load = args.load
#         init_ds = Stacked_MNIST(root, load=load, source_root=root,
#                                 train=True, num_channels = args.data.channels,
#                                 imageSize=args.data.image_size,
#                                 device=args.device)

#     if dataset_transfer_tag == DATASET_STACKEDMNIST:
#         root = os.path.join(args.data_dir, 'mnist')
#         saved_file = os.path.join(root, "data.pt")
#         load = os.path.exists(saved_file)
#         load = args.load
#         final_ds = Stacked_MNIST(root, load=load, source_root=root,
#                                 train=True, num_channels = args.data.channels,
#                                 imageSize=args.data.image_size,
#                                 device=args.device)
#         mean_final = torch.tensor(0.)
#         var_final = torch.tensor(1.*10**3)

#     # EMNIST DATASET
#     if dataset_tag == DATASET_EMNIST:
#         root = os.path.join(args.data_dir, 'EMNIST')
#         saved_file = os.path.join(root, "data.pt")
#         load = os.path.exists(saved_file)
#         load = args.load
#         init_ds = EMNIST(root, load=load, source_root=root,
#                                 train=True, num_channels = args.data.channels,
#                                 imageSize=args.data.image_size,
#                                 device=args.device)

#     if dataset_transfer_tag == DATASET_EMNIST:
#         root = os.path.join(args.data_dir, 'EMNIST')
#         saved_file = os.path.join(root, "data.pt")
#         load = os.path.exists(saved_file)
#         load = args.load
#         final_ds = EMNIST(root, load=load, source_root=root,
#                                 train=True, num_channels = args.data.channels,
#                                 imageSize=args.data.image_size,
#                                 device=args.device)
#         mean_final = torch.tensor(0.)
#         var_final = torch.tensor(1.*10**3)

#     # FINAL (GAUSSIAN) DATASET (if no transfer)
#     if not args.transfer:
#         final_ds, mean_final, var_final = None, None, None
#     if not(args.transfer) and split == 'train':
#         if args.adaptive_mean:
#             NAPPROX = 100
#             # TODO: this batch size causes error when it can not be devided by the datasize
#             vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX)))[0]
#             mean_final = vec.mean()
#             mean_final = vec[0] * 0 + mean_final
#             var_final = eval(args.var_final)
#             final_ds = None
#         elif args.final_adaptive:
#             NAPPROX = 100
#             vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX)))[0]
#             mean_final = vec.mean(axis=0)
#             var_final = vec.var()
#             final_ds = None
#         else:
#             mean_final = eval(args.mean_final)
#             var_final = eval(args.var_final)
#             final_ds = None

#     return init_ds, final_ds, mean_final, var_final


# Logger
# --------------------------------------------------------------------------------

LOGGER = "LOGGER"
LOGGER_PARAMS = "LOGGER_PARAMS"

CSV_TAG = "CSV"
NOLOG_TAG = "NONE"


def get_logger(args, name):
    logger_tag = getattr(args, LOGGER)

    if logger_tag == CSV_TAG:
        kwargs = {"directory": args.CSV_log_dir, "name": name}
        return CSVLogger(**kwargs)

    if logger_tag == NOLOG_TAG:
        return Logger()
