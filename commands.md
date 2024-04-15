
# COMM20
## codes for mini
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=100 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=mini_dim2_nomarginal_thres dataset.datadir=data/comm20_single_graph_small/ ema_rate=0.99

## codes for comm20
CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn project_name=complete_2edgedim dataset.datadir=data/comm20_old/

## codes for single
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=single_2edgedim dataset.datadir=data/comm20_single/ ema_rate=0.99