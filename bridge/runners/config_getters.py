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

cmp = lambda x: transforms.Compose([*x])


# Model
# --------------------------------------------------------------------------------


def get_graph_models(args, dataset_infos):
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
    if cfg["name"] in ["sbm", "sbm_syn", "comm20", "planar", "ego", "planar_edge_remove", "planar_edge_add"] or "sbm_split" in cfg["name"] :
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
        elif cfg["name"] == "sbm_syn":
            datamodule = SBMSynDataModule(cfg)
        elif "sbm_split" in cfg["name"]:
            datamodule = SBMSynDataModule(cfg)
        elif cfg["name"] == "comm20":
            datamodule = Comm20DataModule(cfg)
        elif cfg["name"] == "ego":
            datamodule = EgoDataModule(cfg)
        elif cfg["name"] == "planar":
            datamodule = PlanarDataModule(cfg)

        dataset_infos = SpectreDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()

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

    elif cfg["name"] in ["qm9", "guacamol", "moses", "qm9_smiles"]:
        if cfg["name"] == "qm9":
            from ..datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9Infos(datamodule=datamodule, cfg=cfg)

        elif cfg["name"] == "qm9_smiles":
            from ..datasets import qm9_smiles_dataset

            datamodule = qm9_smiles_dataset.QM9SmilesDataModule(cfg)
            dataset_infos = qm9_smiles_dataset.QM9SmilesInfos(datamodule=datamodule, cfg=cfg)

        elif cfg["name"] == "guacamol":
            from ..datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.GuacamolInfos(datamodule, cfg)

        elif cfg.name == "moses":
            from ..datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MosesInfos(datamodule, cfg)
        else:
            raise ValueError("Dataset not implemented")

        if cfg.extra_features is not None:
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            domain_features = DummyExtraFeatures()

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["name"]))

    return train_metrics, domain_features, datamodule, dataset_infos


# Optimizer
# --------------------------------------------------------------------------------
def get_optimizers(net_f, net_b, lr, n, N):
    # return torch.optim.Adam(net_f.parameters(), lr=lr), torch.optim.Adam(net_b.parameters(), lr=lr)
    print('the new learning rate is', lr * np.exp(-n/N*10))
    return (
        # The code is using the Adam optimizer from the PyTorch library to
        # optimize the parameters of a neural network model `net_f`. It sets the
        # learning rate `lr`, enables the AMSGrad variant of the Adam optimizer by
        # setting `amsgrad=True`, and applies weight decay regularization with a
        # weight decay factor of `1e-12`.
        torch.optim.Adam(net_f.parameters(), lr=lr, amsgrad=True, weight_decay=1e-12),
        torch.optim.Adam(net_b.parameters(), lr=lr, amsgrad=True, weight_decay=1e-12),
    )




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
