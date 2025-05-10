#!/bin/bash
#SBATCH --job-name=Alien16-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=8 # increase 
#SBATCH --mem=16G # increase 
#SBATCH --time=23:59:59
#SBATCH --output=slurm_logs/repVSplan/slurm-%j.out
#SBATCH --error=slurm_logs/repVSplan/slurm-%j.err

module purge
module load anaconda3/2021.11
conda init bash
source ~/.bashrc

conda activate mrq_gpu 

export WANDB_API_KEY=028b65afa7cbbb082902c650f03bfd3801495a70
export WANDB_MODE=offline

python new_new_main.py --env  Atari-Alien-v5 --seed 2 --kind embeddings --total_timesteps 2000000 --device cuda --zs-dim 16 --za-dim 8 --zsa-dim 16 