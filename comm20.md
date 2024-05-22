# comm20

CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn

CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_iter=10000 project_name=comm20 project_name=100steps num_steps=100


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