## QM9

CUDA_VISIBLE_DEVICES=2 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=3 python main.py dataset=qm9 model=gnn num_steps=10 num_iter=100 n_ipf=1000 project_name=debug_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT_BS8192 limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001 cache_npar=8192 plot_npar=8192 batch_size=8192



## Molecular transfer

CUDA_VISIBLE_DEVICES=1 python main.py dataset=qm9 model=gnn num_steps=10 num_iter=10 n_ipf=10 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9_2_qm9 model=gnn num_steps=10 num_iter=10 n_ipf=10 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001



## SBM

CUDA_VISIBLE_DEVICES=3 python main.py main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python main.py main.py dataset=sbm_large_to_small model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001

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


CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf clean_loss_weight=0.0001 lr=0.001 -->