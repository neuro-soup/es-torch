from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pprint import pprint

import gymnasium as gym
import numpy as np
import torch
import wandb
from gymnasium import VectorizeMode
from jaxtyping import Float
from torch import Tensor

from es_torch.optim import Config as ESConfig, ES
from examples.policies import SimpleMLP, SimpleMLPConfig
from examples.utils import (
    CKPTS,
    ESArgumentHandler,
    ExperimentConfig,
    WandbArgumentHandler,
    WandbConfig,
    reshape_params,
    save_policy,
)


@dataclass
class Config(ExperimentConfig):
    policy: SimpleMLPConfig
    epochs: int
    max_episode_steps: int
    env_seed: int | None
    device: str

    @classmethod
    def default(cls) -> Config:
        """More or less from: https://github.com/openai/evolution-strategies-starter/blob/master/configurations/humanoid.json"""
        return cls(
            es=ESConfig(
                n_pop=1440,
                lr=0.01,
                std=0.02,
                weight_decay=0.005,
                sampling_strategy="antithetic",
                reward_transform="centered_rank",
                seed=42,
            ),
            wandb=WandbConfig(
                project="ES-HalfCheetah",
            ),
            policy=SimpleMLPConfig(
                obs_dim=17,
                act_dim=6,
                hidden_dim=64,
            ),
            epochs=1000,
            max_episode_steps=1000,
            env_seed=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


@torch.inference_mode()
def evaluate_policy_batch(
        env: gym.vector.VectorEnv,
        policy_params_batch: Float[Tensor, "npop params"],
        config: Config,
) -> Float[Tensor, "npop"]:
    npop = env.num_envs
    obs, _ = env.reset(seed=config.env_seed)
    dones = np.zeros(npop, dtype=bool)
    total_rewards = np.zeros(npop, dtype=np.float32)

    base_model = SimpleMLP(config.policy).to("meta")

    params_flat = policy_params_batch.to(config.device)
    params_stacked = reshape_params(params_flat, base_model)

    def call_model(params: dict[str, Tensor], o: Tensor) -> Tensor:
        return torch.func.functional_call(base_model, params, (o,))

    vmapped_call_model = torch.vmap(call_model, in_dims=(0, 0))

    for _ in range(config.max_episode_steps):
        obs = torch.tensor(obs, dtype=torch.float32, device=config.device)
        actions = vmapped_call_model(params_stacked, obs)  # cannot mask done envs due to vmap :/ (I think)
        obs, rewards, terminations, truncations, _ = env.step(actions.cpu().numpy())
        dones = dones | terminations | truncations
        total_rewards += rewards * ~dones
        if dones.all():
            break
    if config.wandb.enabled:
        wandb.log(
            {
                "mean_reward": total_rewards.mean(),
                "max_reward": total_rewards.max(),
            }
        )
    print(f"Mean reward: {total_rewards.mean()} | Max reward: {total_rewards.max()}")
    return torch.tensor(total_rewards)


def train(config: Config) -> torch.Tensor:
    pprint(config)
    if config.wandb.enabled:
        if wandb.run is None:  # might be already initialized from sweep
            wandb.init(
                project=config.wandb.project,
                name=config.wandb.name,
                tags=config.wandb.tags,
                entity=config.wandb.entity,
                config=vars(config),
            )

    env = gym.make_vec("HalfCheetah-v5", num_envs=config.es.n_pop, vectorization_mode=VectorizeMode.ASYNC)
    initial_params = torch.nn.utils.parameters_to_vector(SimpleMLP(config.policy).parameters())
    optim = ES(
        config.es,
        params=initial_params,
        eval_fxn=lambda p: (
            evaluate_policy_batch(
                policy_params_batch=p,
                env=env,
                config=config,
            )
        ),
    )

    for epoch in range(config.epochs):
        optim.step()
        print(f"Epoch {epoch + 1}/{config.epochs}")
        if config.wandb.enabled:
            wandb.log({"epoch": epoch + 1})

    env.close()
    if config.wandb.enabled:
        wandb.finish()
    return optim.params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_episode_steps", type=int, help="Max steps per episode")
    parser.add_argument("--hid", type=int, help="Hidden layer size")
    ESArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = vars(parser.parse_args())
    config = Config.default()
    ESArgumentHandler.update_config(args, config)
    WandbArgumentHandler.update_config(args, config)
    config.epochs = args["epochs"] or config.epochs
    config.max_episode_steps = args["max_episode_steps"] or config.max_episode_steps
    config.policy.hidden_dim = args["hid"] or config.policy.hidden_dim

    if config.wandb.enabled:
        run = wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            tags=config.wandb.tags,
            entity=config.wandb.entity,
            config=vars(config),
        )

    #  TODO write a checkpointing callback or sth
    trained_params = train(config)
    model = SimpleMLP(config.policy)
    torch.nn.utils.vector_to_parameters(trained_params, model.parameters())
    filename = "es_half_cheetah.pt" if not config.wandb.enabled else f"es_half_cheetah_{run.name}.pt"
    CKPTS.mkdir(exist_ok=True)
    save_policy(model=model, config=config.policy, save_path=CKPTS / filename)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
