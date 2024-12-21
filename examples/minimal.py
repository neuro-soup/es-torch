from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Protocol

import torch
from jaxtyping import Float
from torch import Tensor


class Sampler(Protocol):
    def __call__(self, n_pop: int, n_params: int, generator: torch.Generator) -> torch.Tensor: ...


class RewardTransform(Protocol):
    def __call__(self, rewards: Float[Tensor, "npop"]) -> Float[Tensor, "npop"]: ...


type SamplingStrategy = Literal["antithetic", "normal"]
SAMPLING_STRATEGIES: dict[SamplingStrategy, Sampler] = {
    "antithetic": lambda npop, nparams, g: torch.cat([eps := torch.randn((npop // 2, nparams), generator=g), -eps], 0),
    "normal": lambda npop, nparams, g: torch.randn((npop, nparams), generator=g),
}
type RewardTransformStrategy = Literal["centered_rank", "normalized"]
REWARD_TRANSFORMS: dict[RewardTransformStrategy, RewardTransform] = {
    "centered_rank": lambda rewards: (rewards.argsort().argsort() - ((len(rewards) - 1) / 2)) / (len(rewards) - 1),
    "normalized": lambda rewards: (rewards - rewards.mean()) / rewards.std(),
}
type EvalFxn = Callable[[Float[Tensor, "npop params"]], Float[Tensor, "npop"]]


@dataclass
class Config:
    n_pop: int
    lr: float
    std: float
    weight_decay: float
    sampling_strategy: SamplingStrategy
    reward_transform: RewardTransformStrategy
    seed: int
    device: str

    def __post_init__(self) -> None:
        assert 0 <= self.weight_decay < 1, "Weight decay should be in [0, 1)."
        assert self.n_pop % 2 == 0, "Number of workers should be even."


class ES:
    def __init__(self, config: Config, params: Float[Tensor, "params"], eval_fxn: EvalFxn) -> None:
        self.cfg = config
        self.params = params
        self._eval_policies = eval_fxn
        self._get_noise = partial(
            SAMPLING_STRATEGIES[config.sampling_strategy],
            config.n_pop,
            len(params),
            torch.Generator().manual_seed(config.seed),
        )
        self._transform_reward = REWARD_TRANSFORMS[config.reward_transform]

    @torch.inference_mode()
    def step(self) -> None:
        noise = self._get_noise()
        perturbations = self.cfg.std * noise
        rewards = self._eval_policies(self.params.unsqueeze(0) + perturbations)
        rewards = self._transform_reward(rewards)
        gradient = self.cfg.lr / (self.cfg.n_pop * self.cfg.std) * torch.einsum("np,n->p", perturbations, rewards)
        self.params += gradient - self.cfg.lr * self.cfg.weight_decay * self.params
