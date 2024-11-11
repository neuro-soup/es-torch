from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Protocol

import torch
from jaxtyping import Float
from torch import Tensor


class Sampler(Protocol):
    def __call__(self, n_pop: int, n_params: int, generator: torch.Generator) -> torch.Tensor: ...


def get_antithetic_noise(n_pop: int, n_params: int, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn((n_pop // 2, n_params), generator=generator)
    return torch.cat([noise, -noise], dim=0)


def get_normal_noise(n_pop: int, n_params: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn((n_pop, n_params), generator=generator)


type RewardTransform = Callable[[Float[Tensor, "npop"]], Float[Tensor, "npop"]]


def get_centered_rank(rewards: Float[Tensor, "npop"]) -> Float[Tensor, "npop"]:
    return rewards.argsort().argsort().float() / len(rewards) - 0.5


def get_normalized(rewards: Float[Tensor, "npop"]) -> Float[Tensor, "npop"]:
    return (rewards - rewards.mean()) / rewards.std()


type SamplingStrategy = Literal["antithetic", "normal"]
SAMPLING_STRATEGIES: dict[SamplingStrategy, Sampler] = {
    "antithetic": get_antithetic_noise,
    "normal": get_normal_noise,
}

type RewardTransformStrategy = Literal["centered_rank", "normalized"]
REWARD_TRANSFORMS: dict[RewardTransformStrategy, RewardTransform] = {
    "centered_rank": get_centered_rank,
    "normalized": get_normalized,
}


@dataclass
class Config:
    n_pop: int
    lr: float
    std: float
    weight_decay: float
    sampling_strategy: SamplingStrategy
    reward_transform: RewardTransformStrategy
    seed: int

    def __post_init__(self) -> None:
        assert 0 <= self.weight_decay < 1, "Weight decay should be in [0, 1)."
        assert self.n_pop % 2 == 0, "Number of workers should be even."


class ES:
    def __init__(self, config: Config, params: Float[Tensor, "params"]) -> None:
        self._cfg = config
        self.params = params.unsqueeze(0)
        self._get_noise = partial(
            SAMPLING_STRATEGIES[config.sampling_strategy],
            config.n_pop,
            len(params),
            torch.Generator().manual_seed(config.seed),
        )
        self._transform_reward = REWARD_TRANSFORMS[config.reward_transform]
        self._perturbed_params: Float[Tensor, "npop params"] | None = None

    @torch.inference_mode()
    def step(self, rewards: Float[Tensor, "npop"]) -> None:
        rewards = self._transform_reward(rewards)
        gradient = (
            self._cfg.lr / (self._cfg.n_pop * self._cfg.std) * torch.einsum("np,n->p", self._perturbed_params, rewards)
        )
        self.params += gradient - self._cfg.lr * self._cfg.weight_decay * self.params

    @torch.inference_mode()
    def get_perturbed_params(self) -> Float[Tensor, "npop params"]:
        noise = self._get_noise()
        self._perturbed_params = self.params.unsqueeze(0) + self._cfg.std * noise
        return self._perturbed_params.squeeze()

    def get_params(self) -> Float[Tensor, "params"]:
        return self.params.squeeze()
