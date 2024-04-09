# CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_steps=4 num_iter=3 n_ipf=3
CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 n_ipf=10 lr=0.001 gamma_max=0.1 ema=True
# CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=500 n_ipf=3
# CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn
# CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=200 n_ipf=10
