# COS-435-RL-MrQ
Our reimplementation of Mr.Q 


Tentative workflow: 
1. models.py (tentative outline done)
2. mrq_agent.py (by Tuesday)
3. utils and env_preprocessing --> utils 
4. replay_buffer.py - can we simplify? 
5. main.py 


Use typer 
Use wandb 
Write loss functions 


---------
ADROIT GET STARTED: 
Clone + cd into repo 
Run following commands: 
module purge

module load anaconda3/2021.11

conda init bash

source ~/.bashrc

conda create -n mrq_gpu python=3.9 -y

conda activate mrq_gpu

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt

pip install wandb typer

sbatch run_mrq.slurm

To check status: squeue -u netid (max 2 gpus per user so will say PD until it starts, then R when running)
To check output/error: cat slurm-{slurm job id}.out or  cat slurm-{slurm job id}.err

After it finishes, cd to folder and run wandb sync --sync-all

-----
Debugging for dm_control: (should work after this)
pip uninstall dm-control mujoco
pip install --upgrade pip
pip install mujoco==2.3.7
pip install dm-control
-------
Running right now for 4/24 deadline: 

GYM: 

     - Gym-Ant-v4_seed_0_logs (done) 

     - Gym-HalfCheetah-v4_seed_0_logs (done) 

     - Gym-Hopper-v4_seed_0_logs (done, slurm 2431100) 

     - Gym-Humanoid-v4_seed_0_logs (done, slurm 2431104)

     - Walker (done, slurm 2431105)

ATARI: 
    - Atari-Alien-v5 seed 0 (done, slurm 2438320) 

DMC_CONTROL: 
    -Dmc-acrobot-swingup seed 0 500K (done 2433346)

----------Embedding Ablations------
Running base set on Gym Ant now (2456894 - cpu) (started 10PM 5/5)