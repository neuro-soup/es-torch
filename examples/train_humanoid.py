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
from examples.utils import CKPTS, add_es_args, reshape_params, save_policy


@dataclass
class Config:
    es: ESConfig
    policy: SimpleMLPConfig
    use_wandb: bool
    epochs: int
    max_episode_steps: int
    env_seed: int | None
    device: str
    wandb_run_name: str | None = None

    @classmethod
    def default(cls) -> Config:
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
            policy=SimpleMLPConfig(
                obs_dim=17,
                act_dim=6,
                hidden_dim=64,
            ),
            use_wandb=False,
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
        actions = vmapped_call_model(params_stacked, obs)  # cannot mask done envs due to :/ (I think)
        obs, rewards, terminations, truncations, _ = env.step(actions.cpu().numpy())
        dones = dones | terminations | truncations
        total_rewards += rewards * ~dones
        if dones.all():
            break
    if config.use_wandb:
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
    if config.use_wandb:
        if wandb.run is None:  # might be already initialized from sweep
            wandb.init(project="ES-HalfCheetah", config=vars(config), name=config.wandb_run_name)

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
        if config.use_wandb:
            wandb.log({"epoch": epoch + 1})

    env.close()
    if config.use_wandb:
        wandb.finish()
    return optim.params


def main() -> None:
    parser = argparse.ArgumentParser()
    add_es_args(parser)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max_episode_steps", type=int)
    parser.add_argument("--hid", type=int)
    args = vars(parser.parse_args())

    config = Config.default()
    config.epochs = args["epochs"] or config.epochs
    config.max_episode_steps = args["max_episode_steps"] or config.max_episode_steps
    config.es.n_pop = args["npop"] or config.es.n_pop
    config.es.std = args["std"] or config.es.std
    config.es.lr = args["lr"] or config.es.lr
    config.policy.hidden_dim = args["hid"] or config.policy.hidden_dim
    config.es.weight_decay = args["wd"] or config.es.weight_decay
    config.es.sampling_strategy = args["noise_strat"] or config.es.sampling_strategy
    config.es.reward_transform = args["reward_strat"] or config.es.reward_transform
    config.es.seed = args["seed"] or config.es.seed
    config.use_wandb = args["wandb"]
    config.wandb_run_name = args["name"]

    #  TODO write a checkpointing callback or sth
    trained_params = train(config)
    model = SimpleMLP(config.policy)
    torch.nn.utils.vector_to_parameters(trained_params, model.parameters())
    filename = "es_half_cheetah.pt" if config.wandb_run_name is None else f"es_half_cheetah_{config.wandb_run_name}.pt"
    save_policy(
        model=model,
        config=config.policy,
        save_path=CKPTS / filename
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
