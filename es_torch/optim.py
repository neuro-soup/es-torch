from dataclasses import dataclass
from typing import Callable, Literal

import torch
from jaxtyping import Float
from torch import Tensor

type Sampler = Callable[[], Float[Tensor, "npop"]]
type SamplingStrategy = Literal["antithetic", "normal"]
SAMPLING_STRATEGIES: dict[SamplingStrategy, Callable[[int, int, torch.Generator], Sampler]] = {
    "antithetic": lambda n_pop, n_params, gen: lambda: _get_antithetic_noise(n_pop, n_params, generator=gen),
    "normal": lambda n_pop, n_params, gen: lambda: torch.randn((n_pop, n_params), generator=gen),
}

type EvalFxn = Callable[[Float[Tensor, "npop params"]], Float[Tensor, "npop"]]

type RewardTransform = Callable[[Float[Tensor, "npop"]], Float[Tensor, "npop"]]
type RewardTransformStrategy = Literal["centered_rank", "normalized"]
REWARD_TRANSFORMS: dict[RewardTransformStrategy, RewardTransform] = {
    "centered_rank": lambda r: r.argsort().argsort() / len(r) - 0.5,
    "normalized": lambda r: (r - r.mean()) / r.std(),
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
    def __init__(
        self,
        config: Config,
        params: Float[Tensor, "params"],
        eval_fxn: EvalFxn,
    ) -> None:
        self._cfg = config
        self.params = params
        self._eval_policies = eval_fxn
        self._get_noise = SAMPLING_STRATEGIES[config.sampling_strategy](
            config.n_pop, len(params), torch.Generator().manual_seed(config.seed)
        )
        self._transform_reward = REWARD_TRANSFORMS[config.reward_transform]

    @torch.inference_mode()
    def step(self) -> None:
        noise = self._get_noise()
        perturbations = self._cfg.std * noise
        rewards = self._eval_policies(self.params.unsqueeze(0) + perturbations)
        rewards = self._transform_reward(rewards)
        gradient = self._cfg.lr / (self._cfg.n_pop * self._cfg.std) * torch.einsum("np,n->p", perturbations, rewards)
        self.params += gradient - self._cfg.lr * self._cfg.weight_decay * self.params


def _get_antithetic_noise(n_pop: int, n_params: int, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn((n_pop // 2, n_params), generator=generator)
    return torch.cat([noise, -noise], dim=0)
