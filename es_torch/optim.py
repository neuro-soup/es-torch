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
    lr: float
    std: float
    seed: int
    device: str


class ES:
    def __init__(
        self,
        config: Config,
        params: Float[Tensor, "params"],
        sampler: Sampler,
        reward_transform: RewardTransform,
        optim: torch.optim.Optimizer,
        std_schedule: Schedule | None = None,
        lr_schedule: Schedule | None = None,
        rng_state: torch.ByteTensor | None = None,
    ) -> None:
        self.cfg = config
        self.params = params.to(config.device)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)  # CPU: reproducibility & easier serialization
        if rng_state is not None:
            self.generator.set_state(rng_state)
        self._sample: Sampler = sampler
        self._transform_reward: RewardTransform = reward_transform
        self._std_schedule: Schedule = std_schedule or (lambda step: config.std)
        self._lr_schedule: Schedule = lr_schedule or (lambda step: config.lr)
        self._step: int = 0
        self._perturbed_params: Float[Tensor, "npop params"] | None = None
        self._optim = optim
        assert all(pg.get("lr", 1.0) == 1.0 for pg in self._optim.param_groups), "Optim lr must be 1.0; lr is controlled by ES config.lr"

    @torch.inference_mode()
    def step(self, rewards: Float[Tensor, "npop"]) -> None:
        rewards = self._transform_reward(rewards)
        std = self._std_schedule(self._step)
        lr = self._lr_schedule(self._step)
        self._step += 1
        gradient = lr / (self.cfg.npop * std) * torch.einsum("np,n->p", self._perturbed_params, rewards)
        self.params.grad = -gradient  # gradient ascent
        self._optim.step()
        self._optim.zero_grad()

    @torch.inference_mode()
    def get_perturbed_params(self) -> Float[Tensor, "npop params"]:
        noise = self._sample(npop=self.cfg.npop, nparams=len(self.params), g=self.generator).to(self.cfg.device)
        std = self._std_schedule(self._step)
        self._perturbed_params = self.params.unsqueeze(0) + std * noise
        return self._perturbed_params.squeeze()
