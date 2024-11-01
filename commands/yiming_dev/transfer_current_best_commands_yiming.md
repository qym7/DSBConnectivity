
# QM9

## Towards smaller SA

### Current best
python launch.py --name=smiles10kreg00001clean00001polyinc --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=qm9_transfer name=smiles10kreg00001clean00001polyinc num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=polyinc batch_size=5096'

python launch.py --name=smiles2kreg00001clean0polyinc --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=qm9_transfer name=smiles2kreg00001clean0polyinc num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0 rand_time=True sample.time_distortion=polyinc batch_size=5096'

python launch.py --name=smiles2kreg0clean00001polyinc --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=qm9_transfer name=smiles2kreg0clean00001polyinc num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=polyinc batch_size=5096'

python launch.py --name=smiles10kreg00001clean00001polyinc20steps --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=qm9_transfer name=smiles10kreg00001clean00001polyinc20steps num_steps=20 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=polyinc batch_size=5096'

python launch.py --name=smiles10kreg00001clean00001polyinc10steps --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=qm9_transfer name=smiles10kreg00001clean00001polyinc10steps num_steps=10 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=polyinc batch_size=5096'

python launch.py --name=smiles10kreg00001clean0001polyinc --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=qm9_transfer name=smiles10kreg00001clean0001polyinc num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.001 rand_time=True sample.time_distortion=polyinc batch_size=5096'

### Ablations
python launch.py --name=smiles10kreg0001clean0001polyinc --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=qm9_transfer name=smiles10kreg0001clean0001polyinc num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=polyinc batch_size=5096'

## Towards higher SA

### Current best

### Ablations

# Planar

## Remove

### Current best

### Ablations

python launch.py --name=planarr10kreg00001clean00001noise10idfeatall --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=planar_edge_remove name=planarr10kreg001clean00001noise10idfeatall num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all save_every_ipf=2'


## Addition

### Current best

# SBM

## 2 to 3

### Current best

### Ablations

python launch.py --name=sbms10kreg0001clean0001noise10identityfeatall --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=sbm_split name=sbms10kreg001clean0001noise10polyincfeatall num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all'

python launch.py --name=sbms10kreg0001clean0noise10identityfeatall --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=sbm_split name=sbms10kreg001clean0noise10polyincfeatall num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.000 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all'

python launch.py --name=sbms10kreg0clean00001noise10identityfeatall --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=sbm_split name=sbms10kreg0clean00001noise10polyincfeatall num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all'

## 3 to 2

### Current best

### Ablations

python launch.py --name=sbmm2kreg0001clean0001noise10identity --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=sbm_merge name=sbmm2kreg001clean0001noise10polyinc num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128'

python launch.py --name=sbmm10kreg0001clean0001noise10identitynew --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=sbm_merge name=sbmm10kreg001clean0001noise10polyinc num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128'

python launch.py --name=sbmm10kreg0001clean0001noise10identityfeatall --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=sbm_merge name=sbmm10kreg001clean0001noise10polyincfeatall num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all'

# Consistency

python launch.py --name=sbmconsistency --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=sbm_consistency name=sbmconsistency num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all'

python launch.py --name=planarconsistency --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=planar_edge_consistency name=planarconsistency num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all'

python launch.py --name=sbmconsistencylessnoise --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=sbm_consistency name=sbmconsistencylessnoise num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all  save_every_ipf=1'


python launch.py --name=planarconsistencyless --gpus=1 --cpus=30 --command='pip install fcd_torch; pip install pygmtools; cd DSBConnectivity; python main.py +experiment=planar_edge_consistency_less name=planarconsistencyless num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=0.3 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all  save_every_ipf=1'
