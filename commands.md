# debug

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=debug wandb=disabled reg_weight=0.0 next_loss_weight=0.0 num_steps=10 num_iter=10


CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=baseline wandb=disabled reg_weight=0.0 next_loss_weight=0.0


CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9 name=qm9_baseline

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9 name=qm9_baseline_next01 next_loss_weight=0.01

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9 name=qm9_baseline_randtime rand_time=True

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_smiles_baseline




<!-- 
## 0803 Molecular debug



CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=debug num_steps=10 num_iter=10 n_ipf=10 virtual_node=False

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles_less_transfer name=qm9less_01noise_5k_clip05_virtual num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=True noise_level=0.1 grad_clip=0.5

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles_less_transfer name=qm9less_005noise_5k_clip05_virtual num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.05 grad_clip=1.0 virtual_node=True

(TORUN) CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_005noise_5k_clip05_virtual num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.05 grad_clip=1.0 virtual_node=True

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_002noise_5k_clip05_3e4clean num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.02 grad_clip=0.5 next_loss_weight=0.0003

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_002noise_5k_clip05_3e4clean_virtual num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=True noise_level=0.02 grad_clip=0.5 next_loss_weight=0.0003

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_01noise_5k num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_clip01 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.1

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_clip02 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.2

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_clip05_vritual num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=True noise_level=0.1 grad_clip=0.5

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_2k num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_005noise_2k_virtual num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=True noise_level=0.05

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_clip2 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=2

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_clip05 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_3e5clean num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 next_loss_weight=0.00003

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_3e4clean num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 next_loss_weight=0.0003

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=qm9_01noise_5k_virtual_clip num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=True noise_level=0.1

## 0902 initial results

(*) CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles test=true num_steps=50 virtual_node=True virtual_node=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/01noise_5k_clip1_virtual/net_f_92_4999.ckpt final_samples_to_generate=10000 name=test_01noise_5k_clip1_virtual_ipf92_10k_noema

(*) CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles test=true num_steps=50 virtual_node=True virtual_node=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/01noise_5k_clip1_virtual/sample_net_f_92_4999.ckpt final_samples_to_generate=10000 name=test_01noise_5k_clip1_virtual_ipf92_10k

(*) CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles test=true num_steps=50 virtual_node=True virtual_node=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/01noise_5k_clip1_virtual/sample_net_f_73_4999.ckpt final_samples_to_generate=10000 name=test_01noise_5k_clip1_virtual_ipf73_10k

(*) CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles num_steps=50 virtual_node=False  test=True  forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/01noise_5k_clip05/sample_net_f_41_4999.ckpt final_samples_to_generate=10000 name=test_01noise_5k_clip05_ipf41_10k

(*) CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles num_steps=50 virtual_node=False  test=True  forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/01noise_5k_clip05/sample_net_f_68_4999.ckpt final_samples_to_generate=10000 name=test_01noise_5k_clip05_ipf68_10k

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles test=true num_steps=50 virtual_node=True num_iter=10 n_ipf=10 virtual_node=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/01noise_5k_clip1_virtual/net_f_73_4999.ckpt final_samples_to_generate=10000 name=test_01noise_5k_clip1_virtual_10k

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles num_steps=50 num_iter=10 n_ipf=10 virtual_node=False test=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/01noise_5k_clip05/sample_net_f_68_4999.ckpt final_samples_to_generate=10000 name=test_01noise_5k_clip05_ipf68_10k

## 0904 other results - not better


CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles num_steps=50 num_iter=10 n_ipf=10 virtual_node=False test=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/005noise_5k_clip05_virtual/sample_net_f_32_4999.ckpt final_samples_to_generate=30000 name=test32_005noise_5k_clip05_virtual_30k wandb=disabled virtual_node=True


CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles num_steps=50 num_iter=10 n_ipf=10 virtual_node=False test=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/005noise_5k_clip1/sample_net_f_40_4999.ckpt final_samples_to_generate=30000 name=test40_005noise_5k_clip1_30k wandb=disabled


CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles num_steps=50 num_iter=10 n_ipf=10 virtual_node=False test=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/005noise_5k_clip1/sample_net_f_57_4999.ckpt final_samples_to_generate=30000 name=test57_005noise_5k_clip1_30k wandb=disabled

## 0904 - plot the noise schedule

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=noise01_plot num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=noise03_plot num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.3

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=noise10_plots num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=1.0

## 0904 - regularize the rate matrix output with x1

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=noise01_clip05_reg0 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 r1_weight=0.0

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=noise03_clip05_reg001 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.3 grad_clip=0.5 r1_weight=0.01

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise03_clip05_reg0 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.3 grad_clip=0.5 r1_weight=0.0

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise01_clip05_reg1e3 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 r1_weight=0.001  

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=noise01_clip05_reg1e2 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 r1_weight=0.01

## 0905 


CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=noise01_clip05_regpredwithnorm1e2 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 r1_weight=0.01

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=noise01_clip05_regpredwithnorm1e4 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 r1_weight=0.0001

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg0 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 r1_weight=0.0

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg01 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 r1_weight=0.1

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg10 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 r1_weight=10

## 0908

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg01_edgeweight5 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=5.0

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg01_edgeweight3 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=3.0

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg01_edgeweight03 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=0.3

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg01_edgeweight3_virtual num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=True noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=3.0

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=noise01_clip05_l05reg01_edgeweight3 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=3.0

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg02_edgeweight3 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.2 grad_clip=0.5 reg_weight=0.1 edge_weight=3.0

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise01_clip05_l1_5reg03_edgeweight3 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.2 grad_clip=0.5 reg_weight=0.1 edge_weight=3.0

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg005_edgeweight3 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 reg_weight=0.05 edge_weight=3.0

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg02_edgeweight3 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 reg_weight=0.2 edge_weight=3.0

## 0912

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg01_edgeweight5 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=5.0

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg01_edgeweight1 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=1.0

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg01_edgeweight1_virtual num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=True noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=1.0

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg05_edgeweight1 num_steps=50 num_iter=5000 n_ipf=1000 noise_level=0.1 grad_clip=0.5 reg_weight=0.5 edge_weight=1.0

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg0_edgeweight1 num_steps=50 num_iter=5000 n_ipf=1000 noise_level=0.1 grad_clip=0.5 reg_weight=0. edge_weight=1.0

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg0_edgeweight1_clean1e3 num_steps=50 num_iter=5000 n_ipf=1000 noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=1.0 next_loss_weight=0.001

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=noise01_clip05_l1reg0_edgeweight1_clean1e5 num_steps=50 num_iter=5000 n_ipf=1000 noise_level=0.1 grad_clip=0.5 reg_weight=0.1 edge_weight=1.0 next_loss_weight=0.00001

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=noise003_clip05_l1reg01_edgeweight5 num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.03 grad_clip=0.5 reg_weight=0.1 edge_weight=5.0

## QM9

CUDA_VISIBLE_DEVICES=2 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=3 python main.py dataset=qm9 model=gnn num_steps=10 num_iter=100 n_ipf=1000 project_name=debug_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT_BS8192 limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001 batch_size=8192 plot_npar=8192 batch_size=8192

## Molecular transfer

CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=REVERSEto2AUG_0LimitDist_EdgeLoss_FixNitialSample_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=REVERSEto2AUG_0LimitDist_EdgeLoss_FixNitialSample_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=REVERSEto2AUG_FixNitialSample_ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py dataset=qm9 model=gnn num_steps=10 num_iter=10 n_ipf=10 project_name=debug limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=0 python main.py dataset=qm9 model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=RECHECK_ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles  num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT_Virtual_UseEdgeLoss limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001 virtual_node=True

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles test=true num_steps=50 virtual_node=True num_iter=10 n_ipf=10 virtual_node=False forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/noise01_clip05_r1reg01/sample_net_f_133_4999.ckpt final_samples_to_generate=10000 name=01noise_5k_clip01_r1reg01_10k_133ckpt wandb=disabled

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles test=true num_steps=50 virtual_node=True num_iter=10 n_ipf=10 virtual_node=False forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/noise01_clip05_r1reg01/sample_net_f_86_4999.ckpt final_samples_to_generate=10000 name=01noise_5k_clip01_r1reg01_10k_86ckpt wandb=disabled

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles test=true num_steps=50 virtual_node=True num_iter=10 n_ipf=10 virtual_node=False forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/noise01_clip05_r1reg01/sample_net_f_40_4999.ckpt final_samples_to_generate=10000 name=01noise_5k_clip01_r1reg01_10k_40ckpt wandb=disabled

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles test=true num_steps=50 virtual_node=True num_iter=10 n_ipf=10 virtual_node=False forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/noise01_clip05_r1reg01/sample_net_f_20_4999.ckpt final_samples_to_generate=10000 name=01noise_5k_clip01_r1reg01_10k_20ckpt wandb=disabled

# QM9 smiles

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_noclean num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1 next_loss_weight=0.0 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=debug num_steps=50 num_iter=1000 n_ipf=1000 virtual_node=False noise_level=0.1 next_loss_weight=0.0 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_1e4cleancloss_edgeloss num_steps=50 num_iter=1000 n_ipf=1000 virtual_node=False noise_level=0.1 next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py +experiment=qm9_smiles name=qm9_01noise_5k num_steps=50 num_iter=5000 n_ipf=1000 virtual_node=False noise_level=0.05 noise_level=0.1

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_005noise num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.05 noise_level=0.05

CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles name=qm9_01noise_3e4clean num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1 next_loss_weight=0.0003

CUDA_VISIBLE_DEVICES=1 python main.py +experiment=qm9_smiles name=qm9_01noise_virtual num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1 virtual_node=True

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=qm9_01noise_3e5clean num_steps=50 num_iter=2000 n_ipf=1000 virtual_node=False noise_level=0.1 next_loss_weight=0.00003

CUDA_VISIBLE_DEVICES=3 python main.py +experiment=qm9_smiles name=debug num_steps=50 num_iter=20 n_ipf=1000 virtual_node=False noise_level=0.1 next_loss_weight=0.00003

## SBM

CUDA_VISIBLE_DEVICES=3 python main.py main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python main.py main.py dataset=sbm_split model=gnn num_steps=50 num_iter=50 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python main.py main.py dataset=sbm_large_to_small model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

# test
CUDA_VISIBLE_DEVICES=0 python main.py +experiment=qm9_smiles wandb=disabled test=True forward_path=/home/yqin/coding/flow/DSBConnectivity/checkpoints/noise_01_5k_clipping/sample_net_f_51_4999.ckpt dataset.batch_size=2048 final_samples_to_generate=10000 name=qm9_01noise_test_1k

<!-- # comm20

CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_iter=5000 num_steps=50 n_ipf=30

CUDA_VISIBLE_DEVICES=3 python main.py dataset=comm20 model=gnn num_iter=10000 project_name=comm20 project_name=100steps num_steps


CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_iter=10000 project_name=comm20 project_name=marginal

CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_iter=2000 dataset.datadir=comm20_single project_name=overfit_single

CUDA_VISIBLE_DEVICES=1 python main.py dataset=comm20 model=gnn num_iter=100 dataset.datadir=comm20_single project_name=debug wandb=disabled

CUDA_VISIBLE_DEVICES=0 python main.py dataset=comm20 model=gnn num_iter=500 dataset.datadir=comm20_mini project_name=mini_debug wandb=disabled

# sbm - transfer learning

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_transfer model=gnn num_steps=50 num_iter=5000 n_ipf=40 project_name=sbm_transfer

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_transfer model=gnn num_steps=50 num_iter=5000 n_ipf=20 project_name=sbm_transfer_useprevnet

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_transfer_less model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_3_to_2 limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_transfer_less2 model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_2_to_3 limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_transfer_less model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_3_to_2_cos limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_transfer_less model=gnn num_steps=50 num_iter=5000 n_ipf=40 project_name=sbm_transfer_less limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_transfer model=gnn num_steps=50 num_iter=100 n_ipf=20 project_name=debug

# sbm split

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_new_split_veryfew_noise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_new_split limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle_posenc limit_dist=marginal_tf model.positional_encoding=True

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=sbm_split_shuffle limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=REVERSE_ce_loss_less_noise limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split_reverse model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=REVERSE_ce_loss_less_noise_5k limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_less_noise_5k limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_3noise limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_32noise limit_dist=marginal_tf next_loss_weight=0.1

CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split_small model=gnn num_steps=100 num_iter=2000 n_ipf=1000 project_name=SMALL_ce_loss_less_noise_2k limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split_more model=gnn num_steps=100 num_iter=5000 n_ipf=1000 project_name=MORE_ce_loss_less_noise_5k_LessLrF limit_dist=marginal_tf

CUDA_VISIBLE_DEVICES=2 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss limit_dist=marginal_tf


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e2CleanLoss_reverseT limit_dist=marginal_tf next_loss_weight=0.01

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss_DecLR limit_dist=marginal_tf next_loss_weight=0.001

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss_LessLrF limit_dist=marginal_tf next_loss_weight=0.001

CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss_Lr1e5 limit_dist=marginal_tf next_loss_weight=0.001 lr=0.00001


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split_more model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=MORE_ce_loss_2k_1e3CleanLoss_Lr1e5 limit_dist=marginal_tf next_loss_weight=0.001 lr=0.00001


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss limit_dist=marginal_tf next_loss_weight=0.001


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split_more model=gnn num_steps=200 num_iter=2000 n_ipf=1000 project_name=MORE_ce_loss_2k_1e3CleanLoss_Lr1e4 limit_dist=marginal_tf next_loss_weight=0.001 lr=0.0001


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_1e3CleanLoss_TryAGAIN limit_dist=marginal_tf next_loss_weight=0.001


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss_Lr3 limit_dist=marginal_tf next_loss_weight=0.001 lr=0.001

CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss_Lr5 limit_dist=marginal_tf next_loss_weight=0.001 lr=0.00001


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss_Lr4Lr5 limit_dist=marginal_tf next_loss_weight=0.001 lr=0.00001


CUDA_VISIBLE_DEVICES=0 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss_FLr4BLr5 limit_dist=marginal_tf next_loss_weight=0.001 lr=0.00001




CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e5CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.00001 lr=0.00001



CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_0CleanLoss_Baseline_ReverseMetrics limit_dist=marginal_tf next_loss_weight=0.0


CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_0CleanLoss_Baseline_ReverseMetrics limit_dist=marginal_tf next_loss_weight=0.0



CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=50 n_ipf=1000 project_name=ce_loss_debug limit_dist=marginal_tf next_loss_weight=0.0




CUDA_VISIBLE_DEVICES=1 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_1e3CleanLoss limit_dist=marginal_tf next_loss_weight=0.001


CUDA_VISIBLE_DEVICES=3 python main.py  dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=ce_loss_5k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=2 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3 limit_dist=marginal_tf next_loss_weight=0.0 lr=0.001


CUDA_VISIBLE_DEVICES=1 python main.py dataset=sbm_split model=gnn num_steps=50 num_iter=2000 n_ipf=1000 project_name=ce_loss_2k_Lr3_1e4CleanLossRegByT limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=3 python3 main.py dataset=sbm_split model=gnn num_steps=50 num_iter=5000 n_ipf=1000 project_name=sbm_split_best_test_graphs limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001


CUDA_VISIBLE_DEVICES=3 python3 main.py dataset=planar_edge_remove model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=fixed_degree_clw0.0001_lr0.001 limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001

CUDA_VISIBLE_DEVICES=7 python3 main.py dataset=planar_edge_add model=gnn num_steps=50 num_iter=5000 n_ipf=30 project_name=fixed_shortest_path_clw0.0001_lr0.001 limit_dist=marginal_tf next_loss_weight=0.0001 lr=0.001 -->
