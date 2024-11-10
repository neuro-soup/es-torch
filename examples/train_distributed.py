from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import gymnasium as gym
import numpy as np
import ray
import torch
import wandb
from gymnasium import VectorizeMode
from jaxtyping import Float
from torch import Tensor

from es_torch.distrbuted import Config as ESConfig, ES
from examples.policies import SimpleMLP, SimpleMLPConfig
from examples.utils import (
    ESArgumentHandler,
    ExperimentConfig,
    Paths,
    WandbArgumentHandler,
    WandbConfig,
    reshape_params, save_policy,
)


@dataclass
class Config(ExperimentConfig):
    policy: SimpleMLPConfig
    epochs: int
    max_episode_steps: int
    env_seed: int | None
    device: str
    ckpt_every: int = -1
    ckpt_path: str | Path | None = None

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
                project="ES-Humanoid",
            ),
            policy=SimpleMLPConfig(
                obs_dim=348,
                act_dim=17,
                hidden_dim=256,
            ),
            epochs=1000,
            max_episode_steps=1000,
            env_seed=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


@ray.remote
class RayWorker:
    def __init__(self, config: Config, optim: ES) -> None:
        self.config = config
        self.optim = optim
        self.env = gym.make_vec("Humanoid-v5", num_envs=config.es.n_pop, vectorization_mode=VectorizeMode.ASYNC)
        self._policy_batch: Float[Tensor, "npop params"] | None = None

    def collect_rollout(self) -> np.ndarray:
        npop = self.env.num_envs
        obs, _ = self.env.reset(seed=self.config.env_seed)
        dones = np.zeros(npop, dtype=bool)
        total_rewards = np.zeros(npop, dtype=np.float32)

        base_model = SimpleMLP(self.config.policy).to("meta")

        self._policy_batch = self.optim.get_perturbed_params().to(self.config.device)
        params_stacked = reshape_params(self._policy_batch, base_model)

        def call_model(params: dict[str, Tensor], o: Tensor) -> Tensor:
            return torch.func.functional_call(base_model, params, (o,))

        vmapped_call_model = torch.vmap(call_model, in_dims=(0, 0))

        for _ in range(self.config.max_episode_steps):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.device)
            actions = vmapped_call_model(params_stacked, obs)  # cannot mask done envs due to vmap :/ (I think)
            obs, rewards, terminations, truncations, _ = self.env.step(actions.cpu().numpy())
            dones = dones | terminations | truncations
            total_rewards += rewards * ~dones
            if dones.all():
                break
        return total_rewards

    def step(self, rewards: Float[Tensor, "npop"]) -> None:
        self.optim.step(perturbed_params=self._policy_batch, rewards=rewards)


def train(config: Config) -> torch.Tensor:
    initial_params = torch.nn.utils.parameters_to_vector(SimpleMLP(config.policy).parameters())
    optim = ES(config.es, params=initial_params)
    for epoch in range(config.epochs):
        rewards = []
        while len(rewards) < config.es.n_pop:
            available_workers = get_ray_workers(optim=optim, n_jobs=config.es.n_pop - len(rewards))
            rewards.extend([worker.collect_rollout.remote() for worker in available_workers])
        rewards = torch.tensor(ray.get(rewards))
        [worker.step(rewards) for worker in available_workers]
        optim.step(rewards)  # TODO can we parallelize this too?
        if config.wandb.enabled:
            wandb.log(
                {
                    "mean_reward": rewards.mean(),
                    "max_reward": rewards.max(),
                }
            )
        print(f"Mean reward: {rewards.mean()} | Max reward: {rewards.max()}")
    return optim.params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_episode_steps", type=int, help="Max steps per episode")
    parser.add_argument("--hid", type=int, help="Hidden layer size")
    parser.add_argument("--ckpt", type=int, help="Save every N epochs. N<=0 disables saving")
    ESArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = vars(parser.parse_args())
    cfg = Config.default()
    ESArgumentHandler.update_config(args, cfg)
    WandbArgumentHandler.update_config(args, cfg)
    cfg.epochs = args["epochs"] or cfg.epochs
    cfg.max_episode_steps = args["max_episode_steps"] or cfg.max_episode_steps
    cfg.policy.hidden_dim = args["hid"] or cfg.policy.hidden_dim

    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            entity=cfg.wandb.entity,
            config=vars(cfg),
        )

    filename = "humanoid.pt" if not cfg.wandb.enabled else f"humanoid_{run.name}.pt"
    cfg.ckpt_path = Paths.CKPTS / filename
    cfg.ckpt_every = args["ckpt"] or cfg.ckpt_every

    pprint(cfg)

    final_params = train(cfg)

    model = SimpleMLP(cfg.policy)
    torch.nn.utils.vector_to_parameters(final_params, model.parameters())
    fp = cfg.ckpt_path.with_stem(cfg.ckpt_path.stem + "_final")
    save_policy(
        model=model,
        model_config=cfg.policy,
        fp=fp,
    )
    print(f"Saved final checkpoint to {fp}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
