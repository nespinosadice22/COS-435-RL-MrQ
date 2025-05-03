import dataclasses
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import utils
import wandb
from mrq_agent import Agent


class OnlineExperiment:
    def __init__(
        self,
        agent: object,
        env: object,
        eval_env: object,
        logger: object,
        evals: list,
        t: int,
        total_timesteps: int,
        time_passed: float,
        eval_frequency: int,
        eval_eps: int,
        eval_folder: Path,
        project_name: str,
        save_full: bool = False,
        save_freq: int = 1e5,
        save_folder: Path = Path("./checkpoint"),
    ):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.evals = evals
        self.logger = logger

        self.t = t
        self.time_passed = time_passed
        self.start_time = time.time()

        self.total_timesteps = total_timesteps
        self.eval_frequency = eval_frequency
        self.eval_eps = eval_eps

        self.eval_folder = eval_folder
        self.project_name = project_name
        self.save_full = save_full
        self.save_freq = save_freq
        self.save_folder = save_folder

        self.init_timestep = True

    def run(self):
        state = self.env.reset()
        while self.t <= self.total_timesteps:
            # evaluate when the modulo and the “not just-loaded” guard both pass
            if self.t % self.eval_frequency == 0 and not (
                self.t == 0 and self.init_timestep
            ):
                self.maybe_evaluate()

            if (
                self.save_full
                and self.t % self.save_freq == 0
                and not self.init_timestep
            ):
                self.save_experiment()

            action = self.agent.select_action(np.array(state))
            if action is None:
                action = self.env.action_space.sample()

            next_state, reward, terminated, truncated = self.env.step(action)
            self.agent.replay_buffer.add(
                state, action, next_state, reward, terminated, truncated
            )
            state = next_state

            self.agent.train()

            if terminated or truncated:
                wandb.log(
                    {
                        "train/step": self.t + 1,
                        "train/episode": self.env.ep_num,
                        "train/episode_length": self.env.ep_timesteps,
                        "train/episode_reward": self.env.ep_total_reward,
                    },
                    step=self.t + 1,
                )

                self.logger.log_print(
                    f"Total T: {self.t + 1}, "
                    f"Episode Num: {self.env.ep_num}, "
                    f"Episode T: {self.env.ep_timesteps}, "
                    f"Reward: {self.env.ep_total_reward:.3f}"
                )

                state = self.env.reset()

            self.t += 1
            self.init_timestep = False

    # evaluation happens only a certain frequency
    def maybe_evaluate(self):
        total_reward = np.zeros(self.eval_eps)
        for ep in range(self.eval_eps):
            state, terminated, truncated = self.eval_env.reset(), False, False
            while not (terminated or truncated):
                action = self.agent.select_action(
                    np.array(state), use_exploration=False
                )
                state, _, terminated, truncated = self.eval_env.step(action)
            total_reward[ep] = self.eval_env.ep_total_reward

        mean_reward = total_reward.mean()
        elapsed_min = (time.time() - self.start_time + self.time_passed) / 60.0

        wandb.log(
            {
                "eval/step": self.t,
                "eval/avg_reward": mean_reward,
                "eval/elapsed_min": elapsed_min,
            },
            step=self.t,
        )

        self.logger.title(
            f"Evaluation at {self.t} time steps\n"
            f"Average total reward over {self.eval_eps} episodes: {total_reward.mean():.3f}\n"
            f"Total time passed: {round((time.time() - self.start_time + self.time_passed)/60., 2)} minutes"
        )
        self.evals.append(total_reward.mean())

        eval_dir = self.eval_folder / self.project_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_file = eval_dir / "evals.txt"
        np.savetxt(eval_file, self.evals, fmt="%.14f")
        self.init_timestep = False  # AKM: i think we need this here too, NOT CERTAIN

    # minor tidying from original, core logic the same
    def save_experiment(self):
        self.time_passed += time.time() - self.start_time

        var_dict = {
            "t": self.t,
            "eval_frequency": self.eval_frequency,
            "eval_eps": self.eval_eps,
            "time_passed": self.time_passed,
            "np_seed": np.random.get_state(),
            "torch_seed": torch.get_rng_state(),
        }

        # Make sure target folder exists
        save_dir = self.save_folder / self.project_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata and evals
        np.save(save_dir / "exp_var.npy", var_dict)
        np.savetxt(save_dir / "evals.txt", self.evals, fmt="%.14f")

        # Save environments
        pickle.dump(self.env, save_dir.joinpath("env.pickle").open("wb"))
        pickle.dump(self.eval_env, save_dir.joinpath("eval_env.pickle").open("wb"))

        # Save agent
        self.agent.save(save_dir)
        hp_dict = {"hp": dataclasses.asdict(self.agent.hp)}
        np.save(save_dir / "agent_var.npy", hp_dict)

        wandb.log(
            {
                "checkpoint/saved": 1,
                "checkpoint/step": self.t,
            },
            step=self.t,
        )

        self.logger.title("Saved experiment")


# load experiment is the same as original, wandb added
def load_experiment(
    save_folder: Path,
    project_name: str,
    device: torch.device,
    total_timesteps,
    save_experiment,
    save_freq,
    eval_folder: Path,
    log_folder: Path,
):
    # Load experiment settings
    exp_file = save_folder / project_name / "exp_var.npy"
    exp_dict = np.load(exp_file, allow_pickle=True).item()

    # This is not sufficient to guarantee the experiment will run exactly the same,
    # however, it does mean the original seed is not reused.
    np.random.set_state(exp_dict["np_seed"])
    torch.set_rng_state(exp_dict["torch_seed"])
    # Load eval
    evals_file = save_folder / project_name / "evals.txt"
    evals = np.loadtxt(evals_file).tolist()
    # Load envs
    env_path = save_folder / project_name / "env.pickle"
    eval_env_path = save_folder / project_name / "eval_env.pickle"
    env = pickle.load(env_path.open("rb"))
    eval_env = pickle.load(eval_env_path.open("rb"))  # Load agent

    agent_var_file = save_folder / project_name / "agent_var.npy"
    agent_dict = np.load(agent_var_file, allow_pickle=True).item()

    agent = Agent(
        env.obs_shape,
        env.action_dim,
        env.max_action,
        env.pixel_obs,
        env.discrete,
        device,
        env.history,
        agent_dict["hp"],
    )
    agent.load(save_folder / project_name)

    wandb.run.summary["resumed_from_step"] = int(exp_dict["t"])
    wandb.log(
        {"status": "resumed", "resumed_step": exp_dict["t"]}, step=int(exp_dict["t"])
    )

    log_folder.mkdir(parents=True, exist_ok=True)

    logger = utils.Logger(log_folder / f"{project_name}.txt")
    logger.title("Loaded experiment\n" f"Starting from: {exp_dict['t']} time steps.")

    return OnlineExperiment(
        agent,
        env,
        eval_env,
        logger,
        evals,
        exp_dict["t"],
        total_timesteps,
        exp_dict["time_passed"],
        exp_dict["eval_frequency"],
        exp_dict["eval_eps"],
        eval_folder,
        project_name,
        save_experiment,
        save_freq,
        save_folder,
    )
