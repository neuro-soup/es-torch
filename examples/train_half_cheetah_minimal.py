"""Example usage of the `minimal.py` optim that runs on a single machine. Avg reward with provided default config: ~3000
To reproduce, simply run: `python examples/train_half_cheetah_minimal.py`
For examples with the optim from `es_torch`, see `examples/train_humanoid.py` and `examples/train_humanoid_dist.py`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import gymnasium as gym
import numpy as np
import torch
from gymnasium import VectorizeMode
from jaxtyping import Float
from torch import Tensor

from examples.policies import SimpleMLP, SimpleMLPConfig
from examples.utils import reshape_params
from minimal import Config as ESConfig, ES


@dataclass
class Config:
    es: ESConfig
    policy: SimpleMLPConfig
    epochs: int
    max_episode_steps: int
    env_seed: int | None
    ckpt_every: int = -1
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
                seed=42,
                device="cuda" if torch.cuda.is_available() else "cpu",
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

    policy = SimpleMLP(config.policy).to("meta")
    params_flat = policy_params_batch.to(config.es.device)
    params_stacked = reshape_params(params_flat, policy)

    def call_model(params: dict[str, Tensor], o: Tensor) -> Tensor:
        return torch.func.functional_call(policy, params, (o,))

    vmapped_call_model = torch.vmap(call_model, in_dims=(0, 0))

    for _ in range(config.max_episode_steps):
        obs = torch.tensor(obs, dtype=torch.float32, device=config.es.device)
        actions = vmapped_call_model(params_stacked, obs)  # cannot mask done envs due to vmap :/ (I think)
        obs, rewards, terminations, truncations, _ = env.step(actions.cpu().numpy())
        dones = dones | terminations | truncations
        total_rewards += rewards * ~dones
        if dones.all():
            break
    print(f"Mean reward: {total_rewards.mean()} | Max reward: {total_rewards.max()}")
    return torch.tensor(total_rewards)


def train(config: Config) -> torch.Tensor:
    env = gym.make_vec("HalfCheetah-v5", num_envs=config.es.npop, vectorization_mode=VectorizeMode.ASYNC)
    policy = SimpleMLP(config.policy)
    policy.init_weights()
    optim = ES(
        config.es,
        params=torch.nn.utils.parameters_to_vector(policy.parameters()),
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
    return optim.params


def main() -> None:
    cfg = Config.default()
    pprint(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
