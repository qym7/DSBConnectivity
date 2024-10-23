
# QM9

### Ablations

# Planar

### Ablations

python launch.py --name=planar10kreg00001clean00001noise10idfeatall --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=planar name=planar10kreg001clean00001noise10idfeatall num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=identity batch_size=128 model.extra_features=all save_every_ipf=2'


python launch.py --name=planar10kreg00001clean00001noise10polydecfeatallabsorb --gpus=1 --cpus=30 --command='cd DSBConnectivity; python main.py +experiment=planar name=planar10kreg00001clean00001noise10idfeatallabsorb num_steps=50 num_iter=10000 n_ipf=500 virtual_node=False noise_level=1.0 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True sample.time_distortion=polydec batch_size=128 model.extra_features=all save_every_ipf=2 limit_dist=absorbing'

# SBM

### Ablations
