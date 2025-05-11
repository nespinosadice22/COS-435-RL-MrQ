# COS-435-RL-MrQ
-----------------------------------------------------------------------------
OUR REIMPLEMENTATION: 

Within mrq/, this repository includes: 
1. models.py: contains the Encoder, Policy and Value networks 
2. mrq_agent.py: the key ``Agent'' file that trains those networks 
3. new_new_main.py: our third and final version of main, driver code 
4. losses.py: explicitly writes out the losses from the paper (used in mrq_agent.py)
5. hyperparams.py: original hyperparameters from repo 
6. env_preprocessing.py: environment preprocessing logic 
7. buffer.py: replay buffer logic 
8. two_hot.py: reward encoding logic 
9. utils.py: miscellaneous functions 

Some results can be found under mrq/runs (though we did not push most and used wandb logging instead) 

The original results can be found under original_results (from original repo)

-----------------------------------------------------------------------------
HOW TO RUN: 

cd mrq 

python new_new_main.py --env [environment]--seed [seed] --kind [embeddings or repVSplan] --total_timesteps [timesteps] --device [cuda or cpu]  (--use-planning) --zs-dim [zs_dim] --za-dim [za_dim] --zsa-dim [za_dim]  

-----------------------------------------------------------------------------
ADROIT GET STARTED: 

module purge

module load anaconda3/2021.11

conda init bash

source ~/.bashrc

conda create -n mrq_gpu python=3.9 -y

conda activate mrq_gpu

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt

pip install wandb typer

sbatch run_mrq.slurm (or the right slurm file)

To check status: squeue -u netid (max 2 gpus per user so will say PD until it starts, then R when running)

To check output/error: cat slurm-{slurm job id}.out or  cat slurm-{slurm job id}.err

After it finishes, cd to folder and run wandb sync --sync-all

-----------------------------------------------------------------------------
Debugging for dm_control: (should work after this)

pip uninstall dm-control mujoco

pip install --upgrade pip

pip install mujoco==2.3.7

pip install dm-control


