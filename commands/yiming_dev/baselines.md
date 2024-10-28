# DOCKER
# build the docker image
(make sure you're in the repo root directory)
./publish.sh push

# Launch to RCP
# Interactive job
python launch.py  --interactive --name=test --gpus=1 --cpus=5
# Login job
python launch.py --name=NAME_OF_JOB --shm=10 --gpus=1 --cpus=30 --command='python train.py +experiment=debug --dataset=planar'

python launch.py --name=sleep --interactive --shm=10 --gpus=1 --cpus=30 --command='sleep 6000000'
(add '--dry-ruy' to just print our the yaml config file)

# RUNAI
# describe running job
runai describe job <job-name>
(rd <job-name>)
# list all jobs
runai list jobs
(rl)
# delete job
runai delete job <job-name>
(rdel <job-name>)
# get the output of a job
runai logs <job-name>
(rlogs <job-name>)


# delete docker containers (otherwise cannot delete images that are being used by containers)
docker ps -a --filter ancestor=<image>:latest
docker stop <container_id>
docker rm <container_id1> <container_id2> ...
# delete docker images 
docker rmi <image>:latest
(or via desktop app also)
# run a docker container (pick one)
docker run -it --rm <image> /bin/bash
docker run -it --rm <image> /bin/zsh
docker run -it --rm <image> /usr/bin/fish
(- it allows interactive mode with a terminal, --rm removes the container after it is stopped, /bin/bash starts a bash shell in the container)


# KUBERNETES
# run a job with your kube .yaml file
kubectl create -f example.yaml

# HAAS
# ssh into haas
ssh <gaspar_username>@haas001.rcp.epfl.ch

# Runs

## DiGress Baselines

### Train
python launch.py --name=qm9less --gpus=1 --cpus=30 --command='cd DiGress; pip install -e .; cd src; python main.py +experiment=qm9_no_h dataset=qm9_less general.name=qm9less train.batch_size=5012 train.n_epochs=3000'

### Eval

<!-- python launch.py --name=qm91k --gpus=1 --cpus=30 --command='cd DiGress; pip install -e .; cd src; python main.py +experiment=qm9_no_h dataset=qm9_less general.name=qm91k general.test_only=/mnt/lts4/scratch/home/yqin/DiGress/checkpoints/17-25-12-qm9/checkpoints/qm9/epoch954.ckpt general.final_model_samples_to_generate=1000 train.batch_size=5012 general.wandb=disabled' -->

python launch.py --name=qm9less1k --gpus=1 --cpus=30 --command='cd DiGress; pip install -e .; cd src; python main.py +experiment=qm9_no_h dataset=qm9_less general.name=qm9less1k general.test_only=/mnt/lts4/scratch/home/yqin/DiGress/checkpoints/09-07-51-qm9less/checkpoints/qm9less/last-v1.ckpt general.final_model_samples_to_generate=1000 train.batch_size=5012 general.wandb=disabled general.given_smiles=/mnt/lts4/scratch/home/yqin/DSBConnectivity/data/qm9/qm9_pyg_greater/raw/test.csv general.generated_path=/mnt/lts4/scratch/home/yqin/DiGress/important_results/10-51-47-qm9ft1k/samples.pkl'

python launch.py --name=qm9ft1k --gpus=1 --cpus=30 --command='cd DiGress; pip install -e .; cd src; python main.py +experiment=qm9_no_h dataset=qm9_less general.name=qm9ft1k general.test_only=/mnt/lts4/scratch/home/yqin/DiGress/checkpoints/qm9_finetune_resume/last.ckpt general.final_model_samples_to_generate=1000 general.wandb=disabled general.given_smiles=/mnt/lts4/scratch/home/yqin/DSBConnectivity/data/qm9/qm9_pyg_greater/raw/test.csv general.generated_path=/mnt/lts4/scratch/home/yqin/DiGress/important_results/13-57-37-qm9less1k/samples.pkl'

<!-- python launch.py --name=qm910k --gpus=1 --cpus=30 --command='cd DiGress; pip install -e .; cd src; python main.py +experiment=qm9_no_h dataset=qm9_less general.name=qm910k general.test_only=/mnt/lts4/scratch/home/yqin/DiGress/checkpoints/17-25-12-qm9/checkpoints/qm9/epoch954.ckpt train.batch_size=5012 general.final_model_samples_to_generate=10000 general.wandb=disabled' -->

python launch.py --name=qm9less10k --gpus=1 --cpus=30 --command='cd DiGress; pip install -e .; cd src; python main.py +experiment=qm9_no_h dataset=qm9_less general.name=qm9less10k general.test_only=/mnt/lts4/scratch/home/yqin/DiGress/checkpoints/09-07-51-qm9less/checkpoints/qm9less/last-v1.ckpt general.final_model_samples_to_generate=10000 train.batch_size=5012 general.wandb=disabled general.given_smiles=/mnt/lts4/scratch/home/yqin/DSBConnectivity/data/qm9/qm9_pyg_greater/raw/test.csv general.generated_path=/mnt/lts4/scratch/home/yqin/DiGress/important_results/11-08-53-qm9ft10k/samples.pkl'

python launch.py --name=qm9ft10k --gpus=1 --cpus=30 --command='cd DiGress; pip install -e .; cd src; python main.py +experiment=qm9_no_h dataset=qm9_less general.name=qm9ft10k general.test_only=/mnt/lts4/scratch/home/yqin/DiGress/checkpoints/qm9_finetune_resume/last.ckpt general.final_model_samples_to_generate=10000 general.wandb=disabled general.given_smiles=/mnt/lts4/scratch/home/yqin/DSBConnectivity/data/qm9/qm9_pyg_greater/raw/test.csv general.generated_path=/mnt/lts4/scratch/home/yqin/DiGress/important_results/16-45-21-qm9less10k/samples.pkl'

## QM9 Transfer evaluation - todo

### Different time sampler

python launch.py --name=qm9evalpolyinc --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalpolyinc num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=polyinc'

python launch.py --name=qm9evalpolydec --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalpolydec num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=polydec'

python launch.py --name=qm9evalid --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalid num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=identity'

python launch.py --name=qm9evalcos --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalcos num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=cos'

python launch.py --name=qm9evalrevcos --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalrevcos num_steps=50 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=revcos'
### Different number of steps



python launch.py --name=qm9evalpolyinc500steps --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalpolyinc500steps num_steps=500 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=polyinc'

python launch.py --name=qm9evalpolyinc100steps --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalpolyinc100steps num_steps=100 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=polyinc'

python launch.py --name=qm9evalpolyinc20steps --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalpolyinc20steps num_steps=20 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=polyinc'


python launch.py --name=qm9evalpolyinc2steps --shm=10 --gpus=1 --cpus=40 --command='pip install fcd_torch; pip install pygmtools; python main.py +experiment=qm9_smiles name=qm9evalpolyinc2steps num_steps=2 num_iter=2000 n_ipf=500 virtual_node=False noise_level=0.1 reg_weight=0.0001 clean_loss_weight=0.0001 rand_time=True batch_size=5096 wandb=disabled test=True sample_checkpoint_f=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_f_20_9999.ckpt sample_checkpoint_b=/mnt/lts4/scratch/home/yqin/DSBConnectivity/checkpoints/qm9transfer/17-31-34/checkpoints/sample_net_b_20_9999.ckpt final_samples_to_generate=10000 sample.time_distortion=polyinc'

