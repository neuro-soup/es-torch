import argparse
import multiprocessing as mp
from dataclasses import asdict

import wandb

from examples.train_humanoid import Config, train
from examples.utils import flatten_dict

# hyperband
FIRST_EVAL_EPOCH = 25
NUM_BRACKETS = 6
ETA = 2  # keep 1/eta runs each bracket
brackets = [FIRST_EVAL_EPOCH * (ETA**i) for i in range(NUM_BRACKETS)]
MAX_EPOCHS = brackets[-1]
SWEEP_CFG = {
    "method": "bayes",
    "metric": {
        "name": "mean_reward",
        "goal": "maximize",
    },
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 2e-3,
            "max": 0.05,  # TODO higher values would work with lr decay
        },
        "std": {
            "distribution": "uniform",
            "min": 0.001,
            "max": 0.04,
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.03,
        },
        # "sampling_strategy": { # todo centered rank seems to be performing slightly worse (impl. error?)
        #     "values": ["antithetic", "normal"],
        # },
        # "reward_transform": {  # seems to have no effect
        #     "values": ["centered_rank", "normalized"],
        # },
        # "hidden_dim": {
        #     "values": [256, 512],
        # },
        # "seed": {  # checking if seed has an effect
        #     "values": [42, 100, 2021],
        # },
    },
    "early_terminate": {"type": "hyperband", "min_iter": FIRST_EVAL_EPOCH, "eta": ETA},
}


def run_sweep() -> None:
    config = Config.default()
    config.use_wandb = True
    config.epochs = MAX_EPOCHS
    config.es.n_pop = 40
    config.policy.hidden_dim = 256
    config.es.reward_transform = "normalized"
    config.es.sampling_strategy = "antithetic"

    run = wandb.init(project="ES-HalfCheetah")
    config.es.lr = run.config.lr
    config.es.std = run.config.std
    config.es.weight_decay = run.config.weight_decay
    # config.es.sampling_strategy = run.config.sampling_strategy
    # config.es.reward_transform = run.config.reward_transform
    # config.es.seed = run.config.seed
    wandb.config.update(flatten_dict(asdict(config)), allow_val_change=True)

    train(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run or continue a W&B sweep for ES-HalfCheetah")
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing sweep ID to continue. If not provided, a new sweep will be created.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print(f"Training will use these evaluation checkpoints (epochs): {brackets}")
    print(f"Final training length will be {MAX_EPOCHS} epochs")

    args = parse_args()
    sweep_id = wandb.sweep(SWEEP_CFG, project="ES-Humanoid") if args.sweep_id is None else args.sweep_id
    wandb.agent(sweep_id, function=run_sweep)
