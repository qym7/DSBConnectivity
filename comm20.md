# comm20

CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_iter=5000 num_steps=50 n_ipf=30

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

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=20000 n_ipf=30 project_name=sbm_new_split_veryfew_noise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=100 project_name=sbm_new_split_veryfew_noise_lessiter_initboth_morelr_linearinc limit_dist=marginal_tf lr=0.0001 gamma_space=linear_inc

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=100 project_name=sbm_new_split_veryfew_noise_lessiter_initboth_morelr_lineardec limit_dist=marginal_tf lr=0.0001 gamma_space=linear_dec

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=100 project_name=sbm_new_split_veryfew_noise_lessiter_initboth limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=100 num_iter=5000 n_ipf=100 project_name=sbm_new_split_veryfew_noise_5000iter_lesslrforward_linearinc_noupdateoptimizer limit_dist=marginal_tf  gamma_space=linear_inc lr=0.0001

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=sbm_new_split_veryfew_noise_2000iter_lesslrforward_noupdateoptimizer_alwaysupdateEMA limit_dist=marginal_tf  gamma_space=linear_inc lr=0.0001

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=sbm_new_split_veryfew_noise_2000iter_lesslrforward_alwaysupdateEMA_UpdateOptimizer limit_dist=marginal_tf  gamma_space=linear_inc lr=0.0001

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=REVERSE_FewNoise_2000iter_LinearInc limit_dist=marginal_tf  gamma_space=linear_inc


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=5000 n_ipf=100 project_name=REVERSE_FewNoise_5000iter_LinearInc limit_dist=marginal_tf  gamma_space=linear_inc

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=FewNoise_2000iter_LinearInc_lr4_lr7_InitBoth limit_dist=marginal_tf  gamma_space=linear_inc

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=FewNoise_2000iter_LinearInc_DecLR limit_dist=marginal_tf  gamma_space=linear_inc

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=5000 n_ipf=100 project_name=REVERSE_FewNoise_5000iter_LinearInc_DecLR limit_dist=marginal_tf  gamma_space=linear_inc

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=VeryFewNoise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=mse_loss_less_noise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=ce_loss_less_noise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=REVERSE_new_loss_less_noise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=5000 n_ipf=100 project_name=REVERSE_new_loss_less_noise_5k limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=REVERSE_FewNoise_2000iter_LinearInc_DecLR_noEMA limit_dist=marginal_tf  gamma_space=linear_inc ema=False

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=FewNoise_2000iter_LinearInc_lr4_lr7_InitBoth_NoEMA limit_dist=marginal_tf  gamma_space=linear_inc lr=0.0000001 ema=False

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=FewNoise_2000iter_LinearInc_InitBoth_UntilConverge limit_dist=marginal_tf  gamma_space=linear_inc lr=0.0000001

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=100 project_name=sbm_new_split_veryfew_noise_evenlessiter_initboth limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_new_split limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle_posenc limit_dist=marginal_tf model.positional_encoding=True

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle limit_dist=marginal_tf

 wandb=disabled