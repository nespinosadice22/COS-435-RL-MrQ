#!/bin/bash
#SBATCH --job-name=PAlien0  #*******************CHANGE ME EACH TIME 
#SBATCH --partition=gpu            
#SBATCH --qos=gpu-medium           
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=16 # increase 
#SBATCH --mem=16G # *************increase for dmc only (use at 16 otw )
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

#uncomment for dmc only - also see additional things to follow in readme ******************
#export MUJOCO_GL=egl
#unset DISPLAY

export WANDB_API_KEY=028b65afa7cbbb082902c650f03bfd3801495a70
export WANDB_MODE=offline


#python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


#DMC - 500K TIMESTEPS. ATARI - 2M TIMESTEPS. GYM - 1M TIMESTEPS. 
#TO TOGGLE PLANNING ON, USE --use-planning flag at end. thats all you need - no true/false after! Will default to regular if flag is omitted. 
python new_new_main.py --env Atari-Alien-v5 --seed 0  --kind repVSplan --total_timesteps 2000000 --device cuda --use-planning
