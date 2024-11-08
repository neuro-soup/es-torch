from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from es_torch.optim import Config as ESConfig


class Paths:
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
        config.wandb.project = args[cls.project] or config.wandb.project
        config.wandb.name = args[cls.name] or config.wandb.name
        config.wandb.tags = args[cls.tags] or config.wandb.tags
        config.wandb.entity = args[cls.entity] or config.wandb.entity


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


def save_policy(
        model: nn.Module,
        model_config: Any,
        fp: str | Path,
) -> None:
    """Save a policy network with its configuration object to a checkpoint."""
    state = {
        "state_dict": model.state_dict(),
        "config": vars(model_config),
    }
    torch.save(state, fp)


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
    ckpt = torch.load(ckpt_path)
    config = config_class(**ckpt["config"])
    policy = policy_class(config, **kwargs)
    policy.load_state_dict(ckpt["state_dict"])
    return policy


def flatten_dict(d: dict) -> dict:
    """{1: {2: 3}, 4: 5} -> {2: 3, 4: 5}"""
    flat = {k: v for k, v in d.items() if not isinstance(v, dict)}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(flatten_dict(v))
    return flat
