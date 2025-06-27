"""Sweep focused on comparing different optimizers for ES."""

import argparse
from dataclasses import asdict
from pprint import pprint

import wandb

from examples.train_half_cheetah_local import Config, train
from examples.utils import flatten_dict

SWEEP_CFG = {
    "method": "bayes",
    "metric": {
        "name": "mean_reward",
        "goal": "maximize",
    },
    "parameters": {
        # Core ES parameters (fixed for fair comparison)
        "lr": {
            "value": 0.04,  # Fixed to isolate optimizer effects
        },
        "std": {
            "value": 0.025,  # Fixed
        },
        "npop": {
            "value": 30,  # Fixed
        },
        
        # Optimizer selection
        "optim": {
            "values": ["SGD", "Adam", "AdamW", "RMSprop"],
        },
        
        # Optimizer-specific parameters
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 0.1,
        },
        "adam_beta1": {
            "distribution": "uniform",
            "min": 0.8,
            "max": 0.99,
        },
        "adam_beta2": {
            "distribution": "uniform", 
            "min": 0.99,
            "max": 0.999,
        },
        "adam_eps": {
            "distribution": "log_uniform_values",
            "min": 1e-8,
            "max": 1e-4,
        },
        
        # Fixed parameters
        "sampling_strategy": {
            "value": "antithetic",
        },
        "reward_transform": {
            "value": "centered_rank",
        },
        "std_schedule": {
            "value": "constant",
        },
        "lr_schedule": {
            "value": "constant",
        },
        "seed": {
            "values": [42, 123, 456],  # Test a few seeds
        },
    },
}


def run_sweep() -> None:
    config = Config.default()
    config.wandb.enabled = True
    config.epochs = 500  # Shorter runs to test more configurations
    
    run = wandb.init(project="ES-HalfCheetah-Optimizers")
    
    # ES core parameters
    config.es.lr = run.config.lr
    config.es.std = run.config.std
    config.es.npop = run.config.npop
    config.es.seed = run.config.seed
    
    # Strategy parameters
    config.sampling_strategy = run.config.sampling_strategy
    config.reward_transform = run.config.reward_transform
    config.std_schedule = run.config.std_schedule
    config.lr_schedule = run.config.lr_schedule
    config.optim = run.config.optim
    
    # Build optimizer kwargs based on optimizer type
    optim_kwargs = {"weight_decay": run.config.weight_decay}
    
    if config.optim in ["Adam", "AdamW"]:
        optim_kwargs.update({
            "betas": (run.config.adam_beta1, run.config.adam_beta2),
            "eps": run.config.adam_eps,
        })
    
    config.optim_kwargs = optim_kwargs
    
    # Log effective configuration
    wandb.config.update({
        "optimizer": config.optim,
        "optimizer_kwargs": optim_kwargs,
    }, allow_val_change=True)
    
    wandb.config.update(flatten_dict(asdict(config)), allow_val_change=True)
    
    pprint(config)
    
    train(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run optimizer comparison sweep for ES")
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing sweep ID to continue. If not provided, a new sweep will be created.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of runs for this agent (default: 1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.sweep_id is None:
        print("Creating new optimizer comparison sweep...")
        sweep_id = wandb.sweep(SWEEP_CFG, project="ES-HalfCheetah-Optimizers")
        print(f"Created sweep with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"Continuing sweep: {sweep_id}")
    
    wandb.agent(sweep_id, function=run_sweep, count=args.count)