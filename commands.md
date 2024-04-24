
# COMM20
## codes for mini
CUDA_VISIBLE_DEVICES=2 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 model.n_layers=2 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=mini_debug_prev02noise use_prev_net=True dataset.datadir=data/comm20_single_graph_small/ ema_rate=0.99

## codes for comm20
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn project_name=complete_prev dataset.datadir=data/comm20_old/ use_prev_net=True

CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn project_name=complete_noprev dataset.datadir=data/comm20_old/ use_prev_net=False

## codes for single
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=single_2edgedim dataset.datadir=data/comm20_single/ ema_rate=0.99

## add node loss
CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 project_name=mini_nodeloss dataset.datadir=data/comm20_single_graph_small/ ema_rate=0.99
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn project_name=complete_nodeloss dataset.datadir=data/comm20_old/ wandb=online use_prev_net=True
CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 project_name=single_nodeloss dataset.datadir=data/comm20_single/ ema_rate=0.99

## dataset interpolation
CUDA_VISIBLE_DEVICES=0 python main.py dataset=sbm_transfer num_steps=50 num_iter=5000 n_ipf=20 visualize_loader=False project_name=sbm40_transfer_prev_net_var10 use_prev_net=True wandb=online

CUDA_VISIBLE_DEVICES=0 python main.py dataset=sbm_transfer num_steps=50 num_iter=50 n_ipf=20 visualize_loader=False project_name=debug use_prev_net=True visualize_loader=True

## prev net - generation
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=mini_noprevnet_nomarginal dataset.datadir=data/comm20_single_graph_small/ ema_rate=0.99 use_prev_net=False
