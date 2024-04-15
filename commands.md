
# COMM20
## codes for mini
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=100 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=marginal_mini dataset.datadir=data/comm20_single_graph_small/ ema_rate=0.99

## codes for comm20
CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn project_name=complete_marginal dataset.datadir=data/comm20_old/

## codes for single
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=single_marginal_stdgrad dataset.datadir=data/comm20_single// ema_rate=0.99