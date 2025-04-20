import dataclasses
import time

import os 
import numpy as np
import torch
import typer

import env_preprocessing
import utils
import wandb
from experiment import OnlineExperiment, load_experiment as load_exp
from mrq_agent import Agent

# not sure why a dataclass is used, but didn't want to change in case
@dataclasses.dataclass
class Defaults:
    Atari_total_timesteps: int = 25e5
    Atari_eval_frequency: int = 1e5

    Dmc_total_timesteps: int = 5e5
    Dmc_eval_frequency: int = 5e3

    Gym_total_timesteps: int = 1e6
    Gym_eval_frequency: int = 5e3

    def __post_init__(self):
        utils.enforce_dataclass_type(self)


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
    eval_folder: str = "./evals",
    log_folder: str = "./logs",
    save_folder: str = "./checkpoint",
    save_experiment: bool = False,
    save_freq: int = 100_000,
    load_experiment: bool = False,
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

    # logger prints to
    log_folder = f"./logs/{env}_seed_{seed}_logs"
    os.makedirs(log_folder, exist_ok=True)
    logger = utils.Logger(f"{log_folder}/{project_name}.txt")

    # set up GPU if requested
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    # start weights & biases tracking
    wandb_settings = wandb.Settings(mode="offline")
    run = wandb.init(
        name=f"run_seed_{seed}_timesteps_{total_timesteps}",
        project="MRQ-Runs-4-19",
        group = f"{env}",
        mode="offline", 
        entity="ak5005-princeton-university",
        config=locals(),
        dir=log_folder
    )

    # either load or create experiment
    if load_experiment:
        exp = load_exp(
            save_folder,
            project_name,
            device,
            total_timesteps,
            eval_frequency,
            eval_eps,
            save_experiment,
            save_freq,
            eval_folder,
            log_folder
        )
    else:
        env = env_preprocessing.Env(env, seed, eval_env=False)
        eval_env = env_preprocessing.Env(env.env_name, seed + 100, eval_env=True)
        agent = Agent(
            env.obs_shape,
            env.action_dim,
            env.max_action,
            env.pixel_obs,
            env.discrete,
            device,
            env.history,
        )

        exp = OnlineExperiment(
            agent=agent,
            env=env,
            eval_env=eval_env,
            evals=[],
            t=0,
            logger = logger,
            total_timesteps=total_timesteps,
            time_passed=0.0,
            eval_frequency=eval_frequency,
            eval_eps=eval_eps,
            eval_folder=eval_folder,
            project_name=project_name,
            save_full=save_experiment,
            save_freq=save_freq,
            save_folder=save_folder,
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


    exp.run()


if __name__ == "__main__":
    app()
