#!/bin/bash
#SBATCH --job-name=Dmc16_2  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=8 # increase 
#SBATCH --mem=8G # increase 
#SBATCH --time=11:59:59
#SBATCH --output=slurm_logs/repVSplan/slurm-%j.out
#SBATCH --error=slurm_logs/repVSplan/slurm-%j.err

# 1 Load modules/conda env
module purge
module load anaconda3/2021.11
conda init bash
source ~/.bashrc

# activate environment (have to make on adroit first)
conda activate mrq_gpu 

export MUJOCO_GL=egl
unset DISPLAY

export WANDB_API_KEY=028b65afa7cbbb082902c650f03bfd3801495a70
export WANDB_MODE=offline


#python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


python new_new_main.py --env Dmc-visual-ball_in_cup-catch  --seed 2 --kind embeddings --total_timesteps 1000000 --device cuda --zs-dim 16 --za-dim 8 --zsa-dim 16  

