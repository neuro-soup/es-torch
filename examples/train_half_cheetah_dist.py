"""At 1k epochs, the provided config reaches a mean reward of ~3k."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any

import evochi.v1 as evochi
import grpc.aio as grpc
import gymnasium as gym
import numpy as np
import psutil
import torch
import wandb
from gymnasium import VectorizeMode
from jaxtyping import Float
from torch import Tensor, nn

from es_torch.optim import Config as ESConfig, ES
from es_torch.sampling import SAMPLERS
from es_torch.fitness_shaping import TRANSFORMS
from es_torch.schedules import SCHEDULES
from examples.policies import SimpleMLP, SimpleMLPConfig
from examples.utils import (
    ESArgumentHandler,
    ExperimentConfig,
    Paths,
    WandbArgumentHandler,
    WandbConfig,
    create_es,
    reshape_params,
    save_policy,
)


@dataclass
class Config(ExperimentConfig):
    epochs: int
    max_episode_steps: int
    policy: SimpleMLPConfig
    ckpt_every: int | None = None
    ckpt_path: str | Path | None = None
    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    env_seed: int | None = None  # different from ESConfig.seed

    @classmethod
    def default(cls) -> Config:
        """More or less from: https://github.com/openai/evolution-strategies-starter/blob/master/configurations/humanoid.json"""
        return cls(
            epochs=1000,
            max_episode_steps=(max_episode_steps := 1000),
            es=ESConfig(
                npop=100,
                lr=0.04,
                std=0.025,
                seed=123323,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            wandb=WandbConfig(
                project="ES-HalfCheetah",
            ),
            sampling_strategy="antithetic",
            reward_transform="centered_rank",
            std_schedule="constant",
            optim="SGD",
            optim_kwargs={"weight_decay": 0.0025},
            policy=SimpleMLPConfig(
                obs_dim=17,
                act_dim=6,
                hidden_dim=64,
            ),
            env_kwargs=dict(
                id="HalfCheetah-v5",
                max_episode_steps=max_episode_steps,
                vectorization_mode=VectorizeMode.ASYNC,
                render_mode=None,
            ),
        )

    def __post_init__(self) -> None:
        assert bool(self.ckpt_every) == bool(self.ckpt_path), "Both `ckpt_every` and `ckpt_path` must be set or unset."
        assert self.ckpt_every is None or self.ckpt_every > 0, "`ckpt_every` must be a positive integer."


@torch.inference_mode()
def evaluate_policy_batch(
    env: gym.vector.VectorEnv,
    model: nn.Module,
    policy_params_batch: torch.Tensor,
    max_episode_steps: int,
    device: torch.device | str,
    env_seed: int,
    env_kwargs: dict[str, Any] = None,
) -> torch.Tensor:
    npop = policy_params_batch.size(0)
    if env is None or env.num_envs != npop:  # less envs if slice doesn't match configured worker bs (usually rare)
        env_kwargs["num_envs"] = npop
        env = gym.make_vec(**env_kwargs)

    obs, _ = env.reset(seed=env_seed)
    dones = np.zeros(npop, dtype=bool)
    total_rewards = np.zeros(npop, dtype=np.float32)

    model = model.to("meta")
    params_flat = policy_params_batch.to(device)
    stacked_params_dict = reshape_params(params_flat, model)

    def call_model(params: dict[str, torch.Tensor], o: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(model, params, (o,))

    vmapped_call_model = torch.vmap(call_model, in_dims=(0, 0))

    for _ in range(max_episode_steps):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        actions = vmapped_call_model(stacked_params_dict, obs)
        obs, rewards, terminations, truncations, _ = env.step(actions.cpu().numpy())
        total_rewards += rewards * ~dones
        dones = dones | terminations | truncations
        if dones.all():
            break
    return torch.tensor(total_rewards, device=device)


@dataclass
class WorkerState:
    params: torch.Tensor
    rng_state: torch.ByteTensor


class Worker(evochi.Worker[WorkerState]):
    def __init__(self, train_cfg: Config, channel: grpc.Channel, cores: int) -> None:
        super().__init__(channel=channel, cores=cores)
        train_cfg.env_kwargs["num_envs"] = cores

        self.cfg = train_cfg
        self.env = gym.make_vec(**self.cfg.env_kwargs)
        self.policy = SimpleMLP(self.cfg.policy)
        self.optim: ES | None = None
        self.perturbed_params: Float[Tensor, "npop nparams"] | None = None

        if self.cfg.wandb.enabled:
            run = wandb.init(
                id=self.cfg.wandb.id,
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.name,
                tags=self.cfg.wandb.tags,
                entity=self.cfg.wandb.entity,
                resume="must" if self.cfg.wandb.id else "never",
            )
        filename = "cheetah.pt" if not self.cfg.wandb.enabled else f"cheetah_{run.name}.pt"
        self.cfg.ckpt_path = Paths.CKPTS / filename

    def initialize(self) -> WorkerState:
        """First worker initializes the state."""
        initial_params = nn.utils.parameters_to_vector(self.policy.parameters())
        self.optim = create_es(self.cfg, params=initial_params, rng_state=None)
        self.perturbed_params = self.optim.get_perturbed_params()
        return WorkerState(
            params=initial_params.cpu(),
            rng_state=self.optim.generator.get_state(),
        )

    def evaluate(self, epoch: int, slices: list[slice]) -> list[evochi.Eval]:
        """Computes the assigned slice (batch) of the population parameters."""
        rewards = evaluate_policy_batch(
            env=self.env,
            model=self.policy,
            policy_params_batch=self.perturbed_params[[i for s in slices for i in range(s.start, s.stop)]],
            max_episode_steps=self.cfg.max_episode_steps,
            device=self.cfg.es.device,
            env_seed=self.cfg.env_seed,
            env_kwargs=self.cfg.env_kwargs,
        )
        print(
            f"(worker): Epoch {epoch} | Mean reward: {rewards.mean()} | Max reward: {rewards.max()} | Slices: {', '.join([f"{s.start}:{s.stop} ({s.stop - s.start})" for s in slices])}"
        )
        return evochi.Eval.from_flat(slices, rewards.tolist())

    def optimize(self, epoch: int, rewards: list[float]) -> WorkerState:
        """Updates the policy parameters based on the rewards."""
        self.optim.step(torch.tensor(rewards, device=self.cfg.es.device))
        self.perturbed_params = self.optim.get_perturbed_params()  # for the next step
        print(f"epoch {epoch}/{self.cfg.epochs}: mean reward {np.mean(rewards)} | max reward {np.max(rewards)}")
        if self.cfg.wandb.enabled:
            rewards = torch.tensor(rewards)
            wandb.log(
                {
                    "epoch": epoch,
                    "mean_reward": rewards.mean().item(),
                    "max_reward": rewards.max().item(),
                    "min_reward": rewards.min().item(),
                    "std_reward": rewards.std().item(),
                }
            )
        if self.cfg.ckpt_every is not None and epoch % self.cfg.ckpt_every == 0:
            torch.nn.utils.vector_to_parameters(self.optim.params, self.policy.parameters())
            fp = self.cfg.ckpt_path.with_stem(f"{self.cfg.ckpt_path.stem}_epoch_{epoch}")
            save_policy(self.policy, model_config=self.cfg.policy, fp=fp)
            print(f"Saved checkpoint to {fp}")
        if epoch == self.cfg.epochs:
            print(
                "Reached configured max epochs. If you forgot to set this on the server, you need to cancel the workers manually or they will keep running."
            )
            self.on_stop(cancel=False)
        return WorkerState(
            params=self.optim.params.cpu(),
            rng_state=self.optim.generator.get_state(),
        )

    def on_state_change(self, state: WorkerState) -> None:
        """Called when a newly joined worker receives the shared state to initialize from."""
        if self.optim is None:
            self.optim = create_es(self.cfg, params=state.params, rng_state=state.rng_state)
        self.optim.generator.set_state(self.state.rng_state)
        self.perturbed_params = self.optim.get_perturbed_params()

    def on_stop(self, cancel: bool) -> None:
        if cancel:
            print("Worker's connection was cancelled by server.")
        torch.nn.utils.vector_to_parameters(self.optim.params, self.policy.parameters())
        fp = self.cfg.ckpt_path.with_stem(f"{self.cfg.ckpt_path.stem}_final")
        save_policy(self.policy, model_config=self.cfg.policy, fp=fp)
        print(f"Saved final parameters to {fp}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_episode_steps", type=int, help="Max steps per episode")
    parser.add_argument("--hid", type=int, help="Hidden layer size")
    parser.add_argument("--ckpt", type=int, help="Save every N epochs. N<=0 disables saving")
    parser.add_argument("--render-mode", type=str, help="Oneof: human, rgb_array, None")
    parser.add_argument("--server", type=str, help="IP address of the server", default="localhost:8080")
    parser.add_argument("--bs", type=int, help="Batch size", default=psutil.cpu_count(logical=True))
    ESArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = vars(parser.parse_args())

    cfg = Config.default()

    ESArgumentHandler.update_config(args, cfg)
    WandbArgumentHandler.update_config(args, cfg)
    cfg.epochs = args["epochs"] or cfg.epochs
    cfg.max_episode_steps = args["max_episode_steps"] or cfg.max_episode_steps
    cfg.policy.hidden_dim = args["hid"] or cfg.policy.hidden_dim
    cfg.env_kwargs["render_mode"] = args["render_mode"] or cfg.env_kwargs.get("render_mode")
    cfg.ckpt_every = args["ckpt"]

    pprint(cfg)

    channel = grpc.insecure_channel(args["server"])
    worker = Worker(cfg, channel=channel, cores=args["bs"])
    await worker.start()


if __name__ == "__main__":
    import multiprocessing as mp

    # else we get warnings from gprc; beware if you edit this script while running with this setting, it will crash...
    mp.set_start_method("spawn")
    asyncio.run(main())
