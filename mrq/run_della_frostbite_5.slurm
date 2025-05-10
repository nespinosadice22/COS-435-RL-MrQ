#!/bin/bash
#SBATCH --job-name=Frost64-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=16 # increase 
#SBATCH --mem=32G # increase 
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

python new_new_main.py --env Atari-Frostbite-v5 --seed 0 --kind embeddings --total_timesteps 2000000 --device cuda --zs-dim 64 --za-dim 32 --zsa-dim 64 & 
python new_new_main.py --env Atari-Frostbite-v5 --seed 1 --kind embeddings --total_timesteps 2000000 --device cuda --zs-dim 64 --za-dim 32 --zsa-dim 64  