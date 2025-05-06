"""
fairly different
"""

from pathlib import Path

import numpy as np
import torch
import typer

import env_preprocessing
import utils
import wandb
from experiment import Experiment
from mrq_agent import Agent
import time 

# not sure why a dataclass is used, but didn't want to change in case
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


# consider using typer.Option(..., help="...") (self-documenting CLI)
@app.command()
def main(
    env: str = "Gym-HalfCheetah-v4",
    seed: int = 0,
    total_timesteps: int = typer.Option(-1, "--total_timesteps"),
    device: str = "cuda",
    eval_frequency: int = -1,
    eval_eps: int = 10,
    project_name: str = "",
    eval_folder: Path = Path("./evals"),
    log_folder: Path = Path("./logs"),
    save_folder: Path = Path("./checkpoint"),
    save_experiment: bool = False,
    save_freq: int = 100_000,
    load_experiment: bool = False,
    zs_dim: int = typer.Option(
        512, "--zs-dim", help="Dimensionality of the state embedding"
    ),
    za_dim: int = typer.Option(
        256, "--za-dim", help="Dimensionality of the action embedding"
    ),
    zsa_dim: int = typer.Option(
        512, "--zsa-dim", help="Dimensionality of the joint embedding"
    ),
):
    config = Defaults()

    # override based on CLU args (same as original main)
    env_type = env.split("-", 1)[0]
    if total_timesteps == -1:
        total_timesteps = config.__dict__[f"{env_type}_total_timesteps"]
    if eval_frequency == -1:
        eval_frequency = config.__dict__[f"{env_type}_eval_frequency"]
    if project_name == "":
        project_name = f"{env}"
    np.random.seed(seed)
    torch.manual_seed(seed)

    # save folders as paths
    eval_folder = Path(eval_folder)
    save_folder = Path(save_folder)

    # logger prints to
    log_folder = Path(f"./logs/embeddings/{env}") / f"{env}_seed_{seed}_zs{zs_dim}_za{za_dim}_zsa{zsa_dim}_timesteps{total_timesteps}"
    log_folder.mkdir(parents=True, exist_ok=True)
    logger = utils.Logger(log_folder / f"{project_name}.txt")

    # set up GPU if requested
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    # start weights & biases tracking
    wandb_settings = wandb.Settings(mode="offline")

    run_config = { 
        "env": env, 
        "seed": seed, 
        "total_timesteps": total_timesteps, 
        "device": str(device), 
        "eval_freq": eval_frequency, 
        "eval_eps": eval_eps, 
        "path": str(log_folder), 
        "zs_dim": zs_dim, 
        "za_dim": za_dim, 
        "zsa_dim": zsa_dim, 
    }
    run = wandb.init(
        name=f"run_{env}_seed_{seed}_zs{zs_dim}_za{za_dim}_zsa{zsa_dim}_timesteps{total_timesteps}",
        project="MRQ-Runs-4-19",
        group=f"{env}-embeddings",
        mode="offline",
        entity="ak5005-princeton-university",
        config=run_config,
        dir=log_folder,
    )
   

    # either load or create experiment
    if load_experiment:
        exp_dir = save_folder / project_name
        exp = Experiment.load_experiment(
            exp_dir,
            device,
            eval_folder,
            log_folder,
        )
    else:
        env = env_preprocessing.Env(env, seed, eval_env=False)
        new_seed = seed + 100
        eval_env = env_preprocessing.Env(env.env_name, new_seed, eval_env=True)
        agent = Agent(
            env.obs_shape,
            env.action_dim,
            env.max_action,
            env.pixel_obs,
            env.discrete,
            device,
            env.history,
            hp={
                "zs_dim": zs_dim,
                "za_dim": za_dim,
                "zsa_dim": zsa_dim,
            },
        )

        # create experiment object
        exp = Experiment(
            agent=agent,
            env=env,
            eval_env=eval_env,
            evals=[],
            t=0,
            logger=logger,
            total_timesteps=total_timesteps,
            time_passed=0.0,
            eval_frequency=eval_frequency,
            eval_eps=eval_eps,
            eval_folder=eval_folder,
            save_folder=save_folder,
            project_name=project_name,
            save_full=save_experiment,
            save_freq=save_freq,
        )

    # kept all the logger logic for now (same as original)
    exp.logger.title("Experiment")
    exp.logger.log_print(f"Algorithm:\t{exp.agent.name}")
    exp.logger.log_print(f"Env:\t\t{exp.env.env_name}")
    exp.logger.log_print(f"Seed:\t\t{exp.env.seed}")

    exp.logger.title("Environment hyperparameters")
    if hasattr(exp.env.env, "hp"):
        exp.logger.log_print(exp.env.env.hp)
    exp.logger.log_print(f"Obs shape:\t\t{exp.env.obs_shape}")
    exp.logger.log_print(f"Action dim:\t\t{exp.env.action_dim}")
    exp.logger.log_print(f"Discrete actions:\t{exp.env.discrete}")
    exp.logger.log_print(f"Pixel observations:\t{exp.env.pixel_obs}")

    exp.logger.title("Agent hyperparameters")
    exp.logger.log_print(exp.agent.hp)
    exp.logger.log_print("-" * 40)
    
    
    start_time = time.time()  
    exp.run()
    duration = time.time() - start_time
    wandb.log({"trial_duration_seconds": duration})
    print(f" (wandb timing) Trial took {duration / 60:.2f} minutes.")
    run.finish() 
   


if __name__ == "__main__":
    app()
