#!/bin/bash
#SBATCH --job-name=mrq_wandb_sweep
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-3               # 4 agents in parallel

# modules/conda env
module purge
module load anaconda3/2021.11
source ~/.bashrc
conda activate mrq_gpu

export WANDB_API_KEY=028b65afa7cbbb082902c650f03bfd3801495a70
export WANDB_MODE=offline

# one agent array per sweep
SWEEP_ID=""  # MUST CHANGE
wandb agent $SWEEP_ID
