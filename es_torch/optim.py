from dataclasses import dataclass
from typing import Protocol

import torch
from jaxtyping import Float
from torch import Tensor


class Sampler(Protocol):
    def __call__(self, npop: int, nparams: int, g: torch.Generator) -> Float[Tensor, "npop nparams"]: ...


class RewardTransform(Protocol):
    def __call__(self, rewards: Float[Tensor, "npop"]) -> Float[Tensor, "npop"]: ...


class Schedule(Protocol):
    def __call__(self, step: int) -> float: ...


@dataclass
class Config:
    npop: int
    std: float
    seed: int
    device: str


class ES:
    def __init__(
        self,
        config: Config,
        sampler: Sampler,
        reward_transform: RewardTransform,
        optim: torch.optim.Optimizer,
        std_schedule: Schedule | None = None,
        rng_state: torch.ByteTensor | None = None,
    ) -> None:
        self.cfg = config
        self._optim = optim
        self.params = list(self._optim.param_groups[0]["params"])[0]
        self.nparams = len(self.params)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)  # CPU: reproducibility & easier serialization
        if rng_state is not None:
            self.generator.set_state(rng_state)
        self._sample: Sampler = sampler
        self._transform_reward: RewardTransform = reward_transform
        self._std_schedule: Schedule = std_schedule or (lambda step: config.std)
        self._step: int = 0
        self._noise: Float[Tensor, "npop params"] | None = None

    def step(self, rewards: Float[Tensor, "npop"]) -> None:
        rewards = self._transform_reward(rewards)
        std = self._std_schedule(self._step)
        self._step += 1
        gradient = 1.0 / (self.cfg.npop * std) * torch.einsum("np,n->p", self._noise, rewards)
        self.params.grad = -gradient
        self._optim.step()
        self._optim.zero_grad()

    def get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self._optim.param_groups[0]["lr"]

    def get_current_std(self) -> float:
        """Get current std from schedule."""
        return self._std_schedule(self._step - 1)  # -1 because we already incremented

    def get_perturbed_params(self) -> Float[Tensor, "npop params"]:
        self._noise = self._sample(npop=self.cfg.npop, nparams=self.nparams, g=self.generator).to(self.cfg.device)
        std = self._std_schedule(self._step)
        return self.params.unsqueeze(0) + std * self._noise
