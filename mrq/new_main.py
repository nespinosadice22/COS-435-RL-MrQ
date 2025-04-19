import dataclasses
import typer
import wandb
import time

import numpy as np
import torch

import env_preprocessing
from experiment import OnlineExperiment, load_experiment
from mrq_agent import Agent
import utils


@dataclasses.dataclass
class Defaults:
    Atari_tot_timesteps: int = 25e5
    Atari_eval_frequency: int = 1e5

    Dmc_tot_timesteps: int = 5e5
    Dmc_eval_frequency: int = 5e3

    Gym_tot_timesteps: int = 1e6
    Gym_eval_frequency: int = 5e3

    def __post_init__(self):
        utils.enforce_dataclass_type(self)


app = typer.Typer()


# consider using typer.Option(..., help="...") (self-documenting CLI)
@app.command()
def main(
    env: str = "Gym-HalfCheetah-v4",
    seed: int = 0,
    tot_timesteps: int = -1,
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

    # override based on CLU args
    env_type = env.split("-", 1)[0]
    if tot_timesteps == -1:
        tot_timesteps = config.__dict__[f"{env_type}_tot_timesteps"]
    if eval_frequency == -1:
        eval_frequency = config.__dict__[f"{env_type}_eval_frequency"]

    if project_name == "":
        project_name = f"MRQ+{env}+{seed}"

    # project name & seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = utils.Logger(f"{log_folder}/{project_name}.txt")

    # set up GPU if requested
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    run = wandb.init(
        name=f"run_seed{seed}_{int(time.time())}",
        project="MRQ-Runs",
        entity="ak5005-princeton-university",
        config=locals(),
    )

    if load_experiment:
        exp = load_experiment(
            save_folder,
            project_name,
            device,
            tot_timesteps,
            eval_frequency,
            eval_eps,
            save_experiment,
            save_freq,
            eval_folder,
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
            tot_timesteps=tot_timesteps,
            time_passed=0.0,
            eval_frequency=eval_frequency,
            eval_eps=eval_eps,
            eval_folder=eval_folder,
            project_name=project_name,
            save_full=save_experiment,
            save_freq=save_freq,
            save_folder=save_folder,
        )

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
