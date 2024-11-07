from __future__ import annotations

import argparse
import multiprocessing as mp
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
    env = gym.make_vec("Humanoid-v5", num_envs=config.es.n_pop, vectorization_mode=VectorizeMode.ASYNC)
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
        if config.ckpt_every > 0 and epoch % config.ckpt_every == 0:
            model = SimpleMLP(config.policy)
            torch.nn.utils.vector_to_parameters(optim.params, model.parameters())
            fp = config.ckpt_path.with_stem(f"{config.ckpt_path.stem}_epoch_{epoch}")
            save_policy(model, model_config=config, fp=fp)
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

    filename = "humanoid.pt" if not cfg.wandb.enabled else f"humanoid_{run.name}.pt"
    cfg.ckpt_path = Paths.CKPTS / filename
    cfg.ckpt_every = args["ckpt"]

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
