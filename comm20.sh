# CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20_debug model=UNET num_steps=4 num_iter=5
# CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=50
CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn

# CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=2000 ema=True project_name=ema_True

# CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_steps=50 num_iter=1000 ema=False project_name=ema_False_real_run
