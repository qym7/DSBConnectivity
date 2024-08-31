## Molecular debug 0830


CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_005noise_5k num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.05

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_01noise_5k num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_2k num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_005noise_2k_virtual num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=True noise_level=0.05



CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_clip2 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=2


CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_clip05 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_3e5clean num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 clean_loss_weight=0.00003

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_3e4clean num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 clean_loss_weight=0.0003

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_virtual_clip num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=True noise_level=0.1

## QM9

CUDA_VISIBLE_DEVICES=2 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=3 python main.py dataset=qm9 model=gnn num_steps=10 num_iter=100 n_ipf=1000 project_name=debug_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT_BS8192 limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001 batch_size=8192 plot_npar=8192 batch_size=8192

## Molecular transfer

CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=REVERSEto2AUG_0LimitDist_EdgeLoss_FixNitialSample_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=REVERSEto2AUG_0LimitDist_EdgeLoss_FixNitialSample_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=REVERSEto2AUG_FixNitialSample_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py dataset=qm9 model=gnn num_steps=10 num_iter=10 n_ipf=10 project_name=debug limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001




CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=RECHECK_ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles  num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT_Virtual_UseEdgeLoss limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001 virtual_node=True

# QM9 smiles

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_noclean num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1 clean_loss_weight=0.0 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=debug num_steps=50 num_iter=1000 n_ipf=1000 virtual_node=False noise_level=0.1 clean_loss_weight=0.0 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_1e4cleancloss_edgeloss num_steps=50 num_iter=1000 n_ipf=1000 virtual_node=False noise_level=0.1 clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_5k num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.05 noise_level=0.1

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_005noise num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.05 noise_level=0.05

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_01noise_3e4clean num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1 clean_loss_weight=0.0003

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_01noise_virtual num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1 virtual_node=True

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=qm9_01noise_3e5clean num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1 clean_loss_weight=0.00003


CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=debug num_steps=50 num_iter=20 n_ipf=1000 virtual_node=False noise_level=0.1 clean_loss_weight=0.00003

## SBM

CUDA_VISIBLE_DEVICES=3 python main.py main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python main.py main.py dataset=sbm_split model=gnn num_steps=50 num_iter=50 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python main.py main.py dataset=sbm_large_to_small model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

# test
CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles wandb=disabled test=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/noise_01_5k_clipping/sample_net_f_51_4999.ckpt dataset.batch_size=2048 final_samples_to_generate=10000 name=qm9_01noise_test_1k




<!-- # comm20

CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_iter=5000 num_steps=50 n_ipf=30

CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_iter=10000 project_name=comm20 project_name=100steps num_steps=


CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_iter=10000 project_name=comm20 project_name=marginal

CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_iter=2000 dataset.datadir=comm20_single project_name=overfit_single

CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_iter=100 dataset.datadir=comm20_single project_name=debug wandb=disabled

CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_iter=500 dataset.datadir=comm20_mini project_name=mini_debug wandb=disabled

# sbm - transfer learning

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_transfer model=gnn num_steps=50 num_iter=5000 n_ipf=40 project_name=sbm_transfer

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_transfer model=gnn num_steps=50 num_iter=5000 n_ipf=20 project_name=sbm_transfer_useprevnet

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_transfer_less model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_3_to_2 limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_transfer_less2 model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_2_to_3 limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_transfer_less model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_3_to_2_cos limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_transfer_less model=gnn num_steps=50 num_iter=5000 n_ipf=40 project_name=sbm_transfer_less limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_transfer model=gnn num_steps=50 num_iter=100 n_ipf=20 project_name=debug

# sbm split

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_new_split_veryfew_noise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_new_split limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle_posenc limit_dist=marginal_tf model.positional_encoding=True

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=REVERSE_ce_loss_less_noise limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=REVERSE_ce_loss_less_noise_5k limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_less_noise_5k limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_3noise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_32noise limit_dist=marginal_tf clean_loss_weight=0.1

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split_small model=gnn num_steps=100 num_iter=2000 n_ipf=1000 project_name=SMALL_ce_loss_less_noise_2k limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split_more model=gnn num_steps=100 num_iter=5000 n_ipf=1000 project_name=MORE_ce_loss_less_noise_5k_LessLrF limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e2CleanLoss_reverseT limit_dist=marginal_tf clean_loss_weight=0.01

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss_DecLR limit_dist=marginal_tf clean_loss_weight=0.001

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss_LessLrF limit_dist=marginal_tf clean_loss_weight=0.001

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss_Lr1e5 limit_dist=marginal_tf clean_loss_weight=0.001 lr=0.00001


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split_more model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=MORE_ce_loss_2k_1e3CleanLoss_Lr1e5 limit_dist=marginal_tf clean_loss_weight=0.001 lr=0.00001


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss limit_dist=marginal_tf clean_loss_weight=0.001


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split_more model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=MORE_ce_loss_2k_1e3CleanLoss_Lr1e4 limit_dist=marginal_tf clean_loss_weight=0.001 lr=0.0001


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss_TryAGAIN limit_dist=marginal_tf clean_loss_weight=0.001


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss_Lr3 limit_dist=marginal_tf clean_loss_weight=0.001 lr=0.001

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss_Lr5 limit_dist=marginal_tf clean_loss_weight=0.001 lr=0.00001


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss_Lr4Lr5 limit_dist=marginal_tf clean_loss_weight=0.001 lr=0.00001


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss_FLr4BLr5 limit_dist=marginal_tf clean_loss_weight=0.001 lr=0.00001




CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e5CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.00001 lr=0.00001



CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_0CleanLoss_Baseline_ReverseMetrics limit_dist=marginal_tf clean_loss_weight=0.0


CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_0CleanLoss_Baseline_ReverseMetrics limit_dist=marginal_tf clean_loss_weight=0.0



CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=50 n_ipf=1000 project_name=ce_loss_debug limit_dist=marginal_tf clean_loss_weight=0.0




CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss limit_dist=marginal_tf clean_loss_weight=0.001


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3 limit_dist=marginal_tf clean_loss_weight=0.0 lr=0.001


CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=3 python3 main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=sbm_split_best_test_graphs limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python3 main.py dataset=planar_edge_remove model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=fixed_degree_clw0.0001_lr0.001 limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=7 python3 main.py dataset=planar_edge_add model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=fixed_shortest_path_clw0.0001_lr0.001 limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001
