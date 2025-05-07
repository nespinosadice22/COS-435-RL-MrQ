import time
from pathlib import Path
import pickle

import numpy as np
import torch
import typer
import wandb

import env_preprocessing
import utils
from mrq_agent import Agent

#from original 
class Defaults:
    def __init__(self):
        self.Atari_total_timesteps = 25e5
        self.Atari_eval_frequency = 1e5
        self.Dmc_total_timesteps = 5e5
        self.Dmc_eval_frequency = 5e3
        self.Gym_total_timesteps = 1e6
        self.Gym_eval_frequency = 5e3
        utils.enforce_types(self)

app = typer.Typer()

@app.command() 
def main(
    #should always specify these 
    env: str = "Gym-HalfCheetah-v4",
    seed: int = typer.Option(0, "--seed"),
    kind: str = typer.Option("embeddings", "--kind", help="embeddings OR repVSplan"),
    total_timesteps: int = typer.Option(-1, "--total_timesteps"),
    device: str = typer.Option("cuda", "--device"),
    #embedding experiments only 
    zs_dim: int = typer.Option(512, "--zs-dim"),
    za_dim: int = typer.Option(256, "--za-dim"),
    zsa_dim: int = typer.Option( 512, "--zsa-dim"),
    #planning experiments only (will default to false otw!)
    use_planning: bool = typer.Option(False, "--use-planning"), 
):
    #----------set up hyperparams-----------------#
    config = Defaults() 
    domain = env.split("-", 1)[0]
    eval_frequency = config.__dict__[f"{domain}_eval_frequency"]
    eval_eps = 10 
    if total_timesteps == -1:
        total_timesteps = config.__dict__[f"{domain}_total_timesteps"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    #------where to save------------------#
    if kind == "embeddings": 
        project_name = "MRQ-Embeddings-Only"
        run_dir = f"runs/embeddings/{env}_seed{seed}_zs{zs_dim}_zsa{zsa_dim}_za{za_dim}_steps{total_timesteps}"
        run_name = f"run_{env}_seed_{seed}_zs{zs_dim}_zsa{zsa_dim}_za{za_dim}_timesteps{total_timesteps}"
    else: 
        project_name = "MRQ-Rep-VS-Plan-Only"
        if use_planning: 
            run_name = f"run_{env}_PLAN_seed_{seed}_timesteps{total_timesteps}_Pdiscount"
            run_dir = f"runs/repVSplan/{env}_seed{seed}_plan_timesteps{total_timesteps}_Pdiscount-nograd"
        else: 
            run_name = f"run_{env}_REP_seed_{seed}_timesteps{total_timesteps}"
            run_dir = f"runs/repVSplan/{env}_seed{seed}_rep_timesteps{total_timesteps}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    #----------set up wandb-----------------#
    run_config = { 
        "env": env, 
        "seed": seed, 
        "total_timesteps": total_timesteps, 
        "device": str(device), 
        "eval_freq": eval_frequency, 
        "eval_eps": eval_eps, 
        "path": str(run_dir), 
        "zs_dim": zs_dim, 
        "za_dim": za_dim, 
        "zsa_dim": zsa_dim, 
        "use_planning": use_planning, 
    }
    run = wandb.init(
        name= run_name,
        project=project_name, 
        group=f"{env}-{kind}",
        mode="offline",
        entity="ak5005-princeton-university",
        config=run_config,
        dir=str(run_dir),
    )
    #---------------Set up--------------------------#
    eval_file = run_dir / "evals.txt"
    fh_eval = eval_file.open("a")
    print(f"RUN DETAILS: env = {str(env)} | seed = {seed} | total_timesteps = {total_timesteps}  | kind = {kind} | device = {device}", file=fh_eval, flush=True)
    if kind == "embeddings": 
        print(f"         | zs = {zs_dim} | zsa = {zsa_dim} | za = {za_dim}", file=fh_eval, flush=True)
    elif use_planning:
        print(f"PLANNING") 
    else: 
        print(f"REP ONLY BASELINE") 
    print(f"{'STEP'}  {'MEAN_R':>12s}", file=fh_eval, flush=True) 

    env = env_preprocessing.Env(env, seed, eval_env=False)
    new_seed = seed + 100
    eval_env = env_preprocessing.Env(env.env_name, new_seed, eval_env=True)
    agent = Agent(env.obs_shape, env.action_dim, env.max_action, env.pixel_obs,env.discrete, 
        device, env.history, hp={"zs_dim": zs_dim,"za_dim": za_dim, "zsa_dim": zsa_dim,
         "use_planning": use_planning})
    
    t_start = 0 
    eval_history = [] 
    start_time = time.time() 
    state = env.reset() 
    #-----------EVAL FUNCTION--------------------#
    def evaluate(step: int): 
        rewards = [] 
        for _ in range(eval_eps): 
            state, terminated, truncated = eval_env.reset(), False, False
            while not (terminated or truncated): 
                action = agent.select_action(np.array(state), use_exploration=False)
                state, _, terminated, truncated = eval_env.step(action) 
            rewards.append(eval_env.ep_total_reward) 
        
        elapsed_time = (time.time() - start_time) / 60 
        mean_reward = float(np.mean(rewards))
        wandb.log({"eval/step": step, "eval/avg_reward": mean_reward})
        print(f"EVAL Step {step}  Reward {mean_reward} TIME (min) {elapsed_time}")
        eval_history.append(mean_reward) 
        print(f"STEP: {step:>10d}, REWARD: {mean_reward:>10.3f} TIME (min) {elapsed_time}", file=fh_eval, flush=True)

    #------------TRAIN!-------------------------------#
    for t in range(t_start, total_timesteps + 1): 
        #check if we should eval 
        if t % eval_frequency == 0 and t != t_start: 
            evaluate(t) 
        #get action 
        action = agent.select_action(np.array(state))
        #too early for replay buffer
        if action is None: 
            action = env.action_space.sample() 
        #take next step and add old one to replay buffer
        next_state, reward, terminated, truncated = env.step(action) 
        agent.replay_buffer.add(state, action, next_state, reward, terminated, truncated)
        state = next_state 
        #train the agent (goes through encoder, value, policy training) + plot logs in wandb (not calculated every t)
        log_data = agent.train() 
        if log_data: 
            wandb.log({k: log_data[k] for k in ("loss/value", "loss/policy", "loss/encoder") if k in log_data},  step=t)
        
        #save / reset on end 
        if terminated or truncated: 
            wandb.log({
                "train/step": t + 1,
                "train/episode": env.ep_num,
                "train/episode_length": env.ep_timesteps,
                "train/episode_reward": env.ep_total_reward
            })
            print(f"TRAIN Total T {t+1:>10d}  Ep Num: {env.ep_num:>4d} Episode T: {env.ep_timesteps:>6d} Reward: {env.ep_total_reward:>10.3f}")

            state = env.reset() 
    #-----------WRAP UP-------------------#
    evaluate(total_timesteps)
    duration = time.time() - start_time 
    wandb.log({"trial_duration_seconds": duration})
    print(f" (wandb timing) Trial took {duration / 60:.2f} minutes.")
    fh_eval.close()
    run.finish() 

if __name__ == "__main__":
    app()

    

