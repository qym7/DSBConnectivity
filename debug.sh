# codes for comm20
CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn project_name=complete_marginal dataset.datadir=data/comm20_old/

# codes for mini
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=marginal_mini dataset.datadir=data/comm20_single_graph_small/

# codes for single
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=single_marginal_stdgrad dataset.datadir=data/comm20_single/

# CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_steps=4 num_iter=3 n_ipf=3
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=single_graph dataset.datadir=data/comm20_single/
CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=mini_graph dataset.datadir=data/comm20_single_graph_small/
CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=5000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=complete_dataset dataset.datadir=data/comm20_old/ thres=1.0
CUDA_VISIBLE_DEVICES=2 python main.py dataset=sbm model=gnn num_steps=50 num_iter=2000 n_ipf=20 lr=0.001 gamma_max=0.1 project_name=sbstart thres=1.0

# CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=500 n_ipf=3
# CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn
# CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=200 n_ipf=10
