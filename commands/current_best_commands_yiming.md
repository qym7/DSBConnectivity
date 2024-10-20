
# QM9

## Towards smaller SA

### Current best
python launch.py --name=2kreg0001clean0001polyinc --gpus=1 --cpus=40 --command='cd DSBConnectivity; python main.py +experiment=qm9_smiles name=2kreg0001clean0001polyinc num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=polyinc batch_size=5096'

### Ablations
python launch.py --name=10kreg0001clean0001polyinc --gpus=1 --cpus=40 --command='cd DSBConnectivity; python main.py +experiment=qm9_smiles name=10kreg0001clean0001polyinc num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=polyinc batch_size=5096'

## Towards higher SA

### Current best

### Ablations
# Planar

## Remove

### Current best

### Ablations
python launch.py --name=planarr2kreg0001clean001noise10identity --gpus=1 --cpus=40 --command='cd DSBConnectivity; python main.py +experiment=planar_edge_remove name=planarr2kreg001clean001noise10polyinc num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.001 rand_time=True sample.time_distortion=identity batch_size=128'

## Addition

### Current best

# SBM

## 2 to 3

### Current best

### Ablations

## 3 to 2

### Current best

### Ablations

python launch.py --name=sbmm2kreg0001clean0001noise10identity --gpus=1 --cpus=40 --command='cd DSBConnectivity; python main.py +experiment=sbm_merge name=sbmm2kreg001clean0001noise10polyinc num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128'

python launch.py --name=sbmm10kreg0001clean0001noise10identity --gpus=1 --cpus=40 --command='cd DSBConnectivity; python main.py +experiment=sbm_merge name=sbmm2kreg001clean0001noise10polyinc num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128'
