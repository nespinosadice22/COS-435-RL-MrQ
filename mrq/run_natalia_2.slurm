#!/bin/bash
#SBATCH --job-name=Alien256-1
#SBATCH --partition=gpu            
#SBATCH --qos=gpu-medium           
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=8 # increase 
#SBATCH --mem=16G # increase 
#SBATCH --time=23:59:59
#SBATCH --output=slurm_logs/repVSplan/slurm-%j.out
#SBATCH --error=slurm_logs/repVSplan/slurm-%j.err

# 1 Load modules/conda env
module purge
module load anaconda3/2021.11
conda init bash
source ~/.bashrc

# activate environment (have to make on adroit first)
conda activate mrq_gpu 

#export MUJOCO_GL=osmesa
#unset DISPLAY

export WANDB_API_KEY=028b65afa7cbbb082902c650f03bfd3801495a70
export WANDB_MODE=offline


#python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

#rep only for humanoid, seeds 1 and 2 
python new_new_main.py --env Atari-Alien-v5 --seed 1 --kind embeddings --total_timesteps 2000000 --device cuda --zs-dim 256 --za-dim 128 --zsa-dim 256
