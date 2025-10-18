# COS-435-RL-MrQ
This project investigates MR.Q, a recently proposed reinforcement learning algorithm designed to achieve general, lightweight, model-free performance across diverse domains such as Atari, Gym, and DMC environments. Unlike domain-specific specialists or heavyweight generalists (DreamerV3, TD-MPC2), MR.Q aims to combine representation learning with a TD3-style backbone to deliver competitive results without excessive compute.

Our contribution is a from-scratch reimplementation of MR.Q with explicit modularity and logging, followed by two key experiments:
- *Representation Only vs. Representation + Planning* — testing whether adding a one-step latent-space planner improves performance.
- *Embedding Dimension Ablations* — examining how reducing the size of learned representations affects generalization across domains.

We evaluate on five representative benchmarks (Atari Alien & Frostbite, Gym Ant & Humanoid, and DMC-Visual Ball in Cup Catch) using multiple seeds, with training/evaluation tracked via Weights & Biases.

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

-----------------------------------------------------------------------------
Debugging for dm_control: (should work after this)

pip uninstall dm-control mujoco

pip install --upgrade pip

pip install mujoco==2.3.7

pip install dm-control


