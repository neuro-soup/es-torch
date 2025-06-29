from dataclasses import dataclass
from typing import Protocol

import torch
from jaxtyping import Float
from torch import Tensor


class Sampler(Protocol):
    def __call__(self, npop: int, nparams: int, g: torch.Generator) -> Float[Tensor, "npop nparams"]: ...


class RewardTransform(Protocol):
    def __call__(self, rewards: Float[Tensor, "npop"]) -> Float[Tensor, "npop"]: ...


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
        rng_state: torch.ByteTensor | None = None,
    ) -> None:
        self.cfg = config
        self._optim = optim
        self.params = list(self._optim.param_groups[0]["params"])[0]
        self.nparams = self.params.numel()
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)  # CPU: reproducibility & easier serialization
        if rng_state is not None:
            self.generator.set_state(rng_state)
        self._sample: Sampler = sampler
        self._transform_reward: RewardTransform = reward_transform
        self.std: float = config.std
        self._noise: Float[Tensor, "npop params"] | None = None

    def step(self, rewards: Float[Tensor, "npop"]) -> None:
        rewards = self._transform_reward(rewards)
        gradient = 1.0 / (self.cfg.npop * self.std) * torch.einsum("np,n->p", self._noise, rewards)
        self.params.grad = -gradient
        self._optim.step()
        self._optim.zero_grad()

    def get_perturbed_params(self) -> Float[Tensor, "npop params"]:
        self._noise = self._sample(npop=self.cfg.npop, nparams=self.nparams, g=self.generator).to(self.cfg.device)
        return self.params.unsqueeze(0) + self.std * self._noise

    @property
    def lr(self) -> float:
        return self._optim.param_groups[0]["lr"]
