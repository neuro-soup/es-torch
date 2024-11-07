from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import torch
from jaxtyping import Float
from torch import Tensor, nn

from es_torch.optim import Config as ESConfig

ROOT = Path(__file__).parent
VIDEOS = ROOT / "videos"
CKPTS = ROOT / "checkpoints"
DATA = ROOT / "data"
LOGS = ROOT / "logs"


@dataclass
class ExperimentConfig:
    es: ESConfig
    wandb: WandbConfig


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    entity: str | None = None


@dataclass
class ESArgumentHandler:
    """Handles argument parsing and configuration for the ES optimizer"""

    population_size: str = "npop"
    std_dev: str = "std"
    learning_rate: str = "lr"
    weight_decay: str = "wd"
    noise_strat: str = "noise"
    reward_strat: str = "reward"
    random_seed: str = "seed"

    # TODO add options in the help
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(f"--{cls.population_size}", type=int, help="Population size")
        parser.add_argument(f"--{cls.std_dev}", type=float, help="Standard deviation of noise")
        parser.add_argument(f"--{cls.learning_rate}", type=float, help="Learning rate")
        parser.add_argument(f"--{cls.weight_decay}", type=float, help="Weight decay")
        parser.add_argument(f"--{cls.noise_strat}", type=str, help="Noise sampling strategy")
        parser.add_argument(f"--{cls.reward_strat}", type=str, help="Reward normalization strategy")
        parser.add_argument(f"--{cls.random_seed}", type=int, help="Seed for noise sampling")

    @classmethod
    def update_config(cls, args: dict[str, Any], config: ExperimentConfig) -> None:
        config.es.n_pop = args[cls.population_size] or config.es.n_pop
        config.es.std = args[cls.std_dev] or config.es.std
        config.es.lr = args[cls.learning_rate] or config.es.lr
        config.es.weight_decay = args[cls.weight_decay] or config.es.weight_decay
        config.es.sampling_strategy = args[cls.noise_strat] or config.es.sampling_strategy
        config.es.reward_transform = args[cls.reward_strat] or config.es.reward_transform
        config.es.seed = args[cls.random_seed] or config.es.seed


@dataclass
class WandbArgumentHandler:
    """Handles argument parsing and configuration for wandb logging"""

    enable: str = "wandb"
    project: str = "project"
    name: str = "name"
    tags: str = "tags"
    entity: str = "entity"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(f"--{cls.enable}", action="store_true", help="Use Weights & Biases logger")
        parser.add_argument(f"--{cls.project}", type=str, help="Name of the Weights & Biases project")
        parser.add_argument(f"--{cls.name}", type=str, help="Name for the Weights & Biases run")
        parser.add_argument(f"--{cls.tags}", nargs="+", help="Tags for the run. Example usage: --tags t1 t2 t3")
        parser.add_argument(f"--{cls.entity}", type=str, help="Wandb entity")

    @classmethod
    def update_config(cls, args: dict[str, Any], config: ExperimentConfig) -> None:
        config.wandb.enabled = args[cls.enable]
        config.wandb.project = args[cls.project]
        config.wandb.name = args[cls.name]
        config.wandb.tags = args[cls.tags]
        config.wandb.entity = args[cls.entity]


def reshape_params(params_flat: Float[Tensor, "npop params"], model: nn.Module) -> dict[str, Float[Tensor, "npop *_"]]:
    npop, _ = params_flat.shape
    param_dict = {}
    param_names = []
    param_shapes = []
    param_sizes = []
    for name, param in model.named_parameters():
        param_names.append(name)
        param_shapes.append(param.shape)
        param_sizes.append(param.numel())
    params_split = torch.split(params_flat, param_sizes, dim=1)
    for name, shape, param in zip(param_names, param_shapes, params_split):
        param_dict[name] = param.view(npop, *shape)
    return param_dict


@torch.inference_mode()
def render_episode(
    model: nn.Module,
    env_id: str,
    output_path: Path,
    max_episode_steps: int = 1000,
    render_fps: int = 30,
    device: str = "cpu",
    **env_kwargs: Any,
) -> float:
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    model = model.to(device).eval()
    frames = []
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(max_episode_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = model(obs_tensor).squeeze(0).cpu().numpy()
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())
        if done or truncated:
            break
    print(f"Episode finished with total reward: {total_reward}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=render_fps)
    env.close()
    return total_reward


def save_policy(
    model: nn.Module,
    config: Any,
    save_path: str | Path,
) -> None:
    """Save a policy network with its configuration object to a checkpoint."""
    state = {
        "state_dict": model.state_dict(),
        "config": vars(config),
    }
    torch.save(state, save_path)


def load_policy(
    ckpt_path: str | Path,
    policy_class: type[nn.Module],
    config_class: type[Any],
    **kwargs,
) -> nn.Module:
    """Load a policy network from a checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file
        policy_class: The class of policy to instantiate
        config_class: The configuration class to use
        **kwargs: Additional arguments passed to the policy constructor
    """
    checkpoint = torch.load(ckpt_path)
    config = config_class(**checkpoint["config"])
    policy = policy_class(config, **kwargs)
    policy.load_state_dict(checkpoint["state_dict"])
    return policy
