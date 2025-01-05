"""Example usage of the es-torch optim on a single machine. Avg reward with provided default config: ~4000.
Note: seed seems to matter a lot...  123323: 4k reward, 42: 1.7k reward
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
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
    ESArgumentHandler,
    ExperimentConfig,
    Paths,
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
    ckpt_every: int | None = None
    ckpt_path: str | Path | None = None

    @classmethod
    def default(cls) -> Config:
        """More or less from: https://github.com/openai/evolution-strategies-starter/blob/master/configurations/humanoid.json"""
        return cls(
            es=ESConfig(
                npop=30,  # original uses 1440, but that's not feasible on a single reasonable machine
                lr=0.04,
                std=0.025,
                weight_decay=0.0025,
                sampling_strategy="antithetic",
                reward_transform="centered_rank",
                seed=123323,
                device="cuda" if torch.cuda.is_available() else "cpu",
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
        )

    def __post_init__(self) -> None:
        assert bool(self.ckpt_every) == bool(self.ckpt_path), "Both `ckpt_every` and `ckpt_path` must be set or unset."
        assert self.ckpt_every is None or self.ckpt_every > 0, "`ckpt_every` must be a positive integer."


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

    params_flat = policy_params_batch.to(config.es.device)
    params_stacked = reshape_params(params_flat, base_model)

    def call_model(params: dict[str, Tensor], o: Tensor) -> Tensor:
        return torch.func.functional_call(base_model, params, (o,))

    vmapped_call_model = torch.vmap(call_model, in_dims=(0, 0))

    for _ in range(config.max_episode_steps):
        obs = torch.tensor(obs, dtype=torch.float32, device=config.es.device)
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
    return torch.tensor(total_rewards, device=config.es.device)


def train(config: Config) -> torch.Tensor:
    env = gym.make_vec("HalfCheetah-v5", num_envs=config.es.npop, vectorization_mode=VectorizeMode.ASYNC)
    policy = SimpleMLP(config.policy)
    policy.init_weights()
    optim = ES(
        config.es,
        params=torch.nn.utils.parameters_to_vector(policy.parameters()),
    )

    for epoch in range(config.epochs):
        rewards = evaluate_policy_batch(env, optim.get_perturbed_params(), config)
        optim.step(rewards)
        print(f"Epoch {epoch + 1}/{config.epochs}")
        if config.wandb.enabled:
            wandb.log({"epoch": epoch + 1})
        if config.ckpt_every is not None and epoch % config.ckpt_every == 0:
            model = SimpleMLP(config.policy)
            torch.nn.utils.vector_to_parameters(optim.params, model.parameters())
            fp = config.ckpt_path.with_stem(f"{config.ckpt_path.stem}_epoch_{epoch}")
            save_policy(model, model_config=config.policy, fp=fp)
            print(f"Saved checkpoint to {fp}")

    env.close()
    if config.wandb.enabled:
        wandb.finish()
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

    filename = "cheetah.pt" if not cfg.wandb.enabled else f"cheetah_{run.name}.pt"
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
    main()
