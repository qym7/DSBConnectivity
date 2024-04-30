
# COMM20
## codes for mini
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=mini_prev_net_no_marginal dataset.datadir=data/comm20_single_graph_small/ ema_rate=0.99 use_prev_net=True

## codes for single
CUDA_VISIBLE_DEVICES=2 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=single_prev dataset.datadir=data/comm20_single/ ema_rate=0.99 use_prev_net=True

## codes for comm20
CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn dataset.datadir=data/comm20_old/ project_name=complete_prev use_prev_net=True
CUDA_VISIBLE_DEVICES=2 python main.py dataset=comm20 model=gnn dataset.datadir=data/comm20_old/ project_name=complete_noprev use_prev_net=False


## add node loss
CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 project_name=mini_nodeloss dataset.datadir=data/comm20_single_graph_small/ ema_rate=0.99
CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn project_name=complete_nodeloss dataset.datadir=data/comm20_old/
CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 project_name=single_nodeloss dataset.datadir=data/comm20_single/ ema_rate=0.99

## dataset interpolation
CUDA_VISIBLE_DEVICES=2 python main.py dataset=sbm_transfer model=gnn num_steps=50 num_iter=10000 n_ipf=20 project_name=debug ema_rate=0.999 visualize_loader=False project_name=sbm40_transfer
