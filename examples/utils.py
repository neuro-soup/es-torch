from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from es_torch.fitness_shaping import TRANSFORMS
from es_torch.optim import Config as ESConfig, ES
from es_torch.sampling import SAMPLERS
from examples.schedules import SCHEDULES, Scheduler


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

    sampling_strategy: str
    reward_transform: str
    std_schedule: str
    lr_schedule: str
    optim: str

    epochs: int
    max_episode_steps: int
    env_seed: int | None

    std_schedule_kwargs: dict
    lr_schedule_kwargs: dict
    optim_kwargs: dict


def create_es(
    cfg: ExperimentConfig,
    params: Float[Tensor, "params"],
    rng_state: torch.ByteTensor | None = None,
) -> tuple[ES, torch.optim.lr_scheduler.LRScheduler, Scheduler]:
    """Factory function to create ES optimizer from experiment config."""
    sampler = SAMPLERS[cfg.sampling_strategy]
    transform = TRANSFORMS[cfg.reward_transform]
    std_schedule = SCHEDULES[cfg.std_schedule](cfg.es.std, **cfg.std_schedule_kwargs)

    optim_class = getattr(torch.optim, cfg.optim)
    params = params.detach().to(cfg.es.device).requires_grad_(True)
    optim = optim_class([params], **cfg.optim_kwargs)

    initial_lr = cfg.optim_kwargs["lr"]  # LambdaLR expects a multiplier, so we divide by initial lr
    lr_schedule_fn = SCHEDULES[cfg.lr_schedule](initial_lr, **cfg.lr_schedule_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: lr_schedule_fn(step) / initial_lr)
    return ES(cfg.es, sampler, transform, optim, rng_state), lr_scheduler, std_schedule


@dataclass
class WandbConfig:
    id: str | None = None
    enabled: bool = False
    project: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    entity: str | None = None


@dataclass
class TrainArgHandler:
    """Handles argument parsing and configuration for the training configuration"""

    population_size: str = "npop"
    learning_rate: str = "lr"
    std_dev: str = "std"
    random_seed: str = "seed"

    noise_strat: str = "noise"
    reward_strat: str = "reward"
    std_schedule: str = "std_schedule"
    lr_schedule: str = "lr_schedule"
    optimizer: str = "optim"

    epochs: str = "epochs"
    max_episode_steps: str = "max_episode_steps"
    env_seed: str = "env_seed"

    std_schedule_kwargs: str = "std_schedule_kwargs"
    lr_schedule_kwargs: str = "lr_schedule_kwargs"
    optimizer_kwargs: str = "optim_kwargs"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(f"--{cls.population_size}", type=int, help="Population size")
        parser.add_argument(f"--{cls.learning_rate}", type=float, help="Learning rate")
        parser.add_argument(f"--{cls.std_dev}", type=float, help="Standard deviation of noise")
        parser.add_argument(f"--{cls.random_seed}", type=int, help="Seed for noise sampling")

        parser.add_argument(f"--{cls.noise_strat}", type=str, help="Noise sampling strategy", choices=SAMPLERS.keys())
        parser.add_argument(f"--{cls.reward_strat}", type=str, help="Reward normalization strategy", choices=TRANSFORMS.keys())
        parser.add_argument(f"--{cls.std_schedule}", type=str, choices=SCHEDULES.keys(), help="Std deviation schedule")
        parser.add_argument(f"--{cls.lr_schedule}", type=str, choices=SCHEDULES.keys(), help="Learning rate schedule")
        parser.add_argument(f"--{cls.optimizer}", type=str, help="Optimizer to use (SGD, Adam, AdamW, etc.)")

        parser.add_argument(f"--{cls.epochs}", type=int, help="Number of training epochs")
        parser.add_argument(f"--{cls.max_episode_steps}", type=int, help="Max steps per episode")
        parser.add_argument(f"--{cls.env_seed}", type=int, help="Environment seed")

        parser.add_argument(f"--{cls.std_schedule_kwargs}", type=str, help="JSON string of std schedule kwargs")
        parser.add_argument(f"--{cls.lr_schedule_kwargs}", type=str, help="JSON string of lr schedule kwargs")
        parser.add_argument(f"--{cls.optimizer_kwargs}", type=str, help="JSON string of optimizer kwargs, e.g. '{\"betas\": [0.9, 0.999]}'")

    @classmethod
    def update_config(cls, args: dict[str, Any], config: ExperimentConfig) -> None:
        config.es.npop = args[cls.population_size] or config.es.npop
        config.es.std = args[cls.std_dev] or config.es.std
        config.es.seed = args[cls.random_seed] or config.es.seed
        if args[cls.learning_rate]:
            config.optim_kwargs["lr"] = args[cls.learning_rate]

        config.sampling_strategy = args[cls.noise_strat] or config.sampling_strategy
        config.reward_transform = args[cls.reward_strat] or config.reward_transform
        config.std_schedule = args[cls.std_schedule] or config.std_schedule
        config.lr_schedule = args[cls.lr_schedule] or config.lr_schedule
        config.optim = args[cls.optimizer] or config.optim

        config.epochs = args[cls.epochs] or config.epochs
        config.max_episode_steps = args[cls.max_episode_steps] or config.max_episode_steps
        config.env_seed = args[cls.env_seed] if args[cls.env_seed] is not None else config.env_seed

        config.std_schedule_kwargs = json.loads(args[cls.std_schedule_kwargs]) if args[cls.std_schedule_kwargs] else config.std_schedule_kwargs
        config.lr_schedule_kwargs = json.loads(args[cls.lr_schedule_kwargs]) if args[cls.lr_schedule_kwargs] else config.lr_schedule_kwargs
        config.optim_kwargs = json.loads(args[cls.optimizer_kwargs]) if args[cls.optimizer_kwargs] else config.optim_kwargs


@dataclass
class WandbArgHandler:
    """Handles argument parsing and configuration for wandb"""

    id: str = "id"
    enable: str = "wandb"
    project: str = "project"
    name: str = "name"
    tags: str = "tags"
    entity: str = "entity"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(f"--{cls.id}", type=str, help="Wandb run ID. Provide only to resume a run.")
        parser.add_argument(f"--{cls.enable}", action="store_true", help="Use Weights & Biases logger")
        parser.add_argument(f"--{cls.project}", type=str, help="Name of the Weights & Biases project")
        parser.add_argument(f"--{cls.name}", type=str, help="Name for the Weights & Biases run")
        parser.add_argument(f"--{cls.tags}", nargs="+", help="Tags for the run. Example usage: --tags t1 t2 t3")
        parser.add_argument(f"--{cls.entity}", type=str, help="Wandb entity")

    @classmethod
    def update_config(cls, args: dict[str, Any], config) -> None:
        config.wandb.id = args[cls.id] or config.wandb.id
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
    fp: Path,
) -> None:
    """Save a policy network with its configuration object to a checkpoint."""
    state = {
        "state_dict": model.state_dict(),
        "config": model_config,
    }
    fp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, fp)


def load_policy(
    ckpt_path: str | Path,
    policy_class: type[nn.Module],
    **kwargs,
) -> nn.Module:
    """Load a policy network from a checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file
        policy_class: The class of policy to instantiate
        **kwargs: Additional arguments passed to the policy constructor
    """
    ckpt = torch.load(ckpt_path)
    policy = policy_class(ckpt["config"], **kwargs)
    policy.load_state_dict(ckpt["state_dict"])
    return policy


def flatten_dict(d: dict) -> dict:
    """{1: {2: 3}, 4: 5} -> {2: 3, 4: 5}"""
    flat = {k: v for k, v in d.items() if not isinstance(v, dict)}
    for _, v in d.items():
        if isinstance(v, dict):
            flat.update(flatten_dict(v))
    return flat


def short_uuid(n: int = 8) -> str:
    return str(uuid.uuid4())[:n]
