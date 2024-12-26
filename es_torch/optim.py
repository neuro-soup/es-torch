from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol

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


@dataclass
class Config:
    npop: int
    lr: float
    std: float
    weight_decay: float
    sampling_strategy: SamplingStrategy
    reward_transform: RewardTransformStrategy
    seed: int
    device: str

    def __post_init__(self) -> None:
        assert 0 <= self.weight_decay < 1, "Weight decay should be in [0, 1)."
        assert (self.npop % 2 == 0) or not (self.sampling_strategy == "antithetic"), "Number of workers should be even."


class ES:
    def __init__(
        self,
        config: Config,
        params: Float[Tensor, "params"],
        rng_state: torch.ByteTensor | None = None,
    ) -> None:
        self.cfg = config
        self.params = params.to(config.device)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)  # CPU: reproducibility & easier sharing
        if rng_state is not None:
            self.generator.set_state(rng_state)
        self._get_noise = partial(
            SAMPLING_STRATEGIES[config.sampling_strategy],
            config.npop,
            len(params),
            self.generator,
        )
        self._transform_reward = REWARD_TRANSFORMS[config.reward_transform]
        self._perturbed_params: Float[Tensor, "npop params"] | None = None

    @torch.inference_mode()
    def step(self, rewards: Float[Tensor, "npop"]) -> None:
        rewards = self._transform_reward(rewards)
        gradient = self.cfg.lr / (self.cfg.npop * self.cfg.std) * torch.einsum("np,n->p", self._perturbed_params, rewards)
        self.params += gradient - self.cfg.lr * self.cfg.weight_decay * self.params

    @torch.inference_mode()
    def get_perturbed_params(self) -> Float[Tensor, "npop params"]:
        noise = self._get_noise().to(self.cfg.device)
        self._perturbed_params = self.params.unsqueeze(0) + self.cfg.std * noise
        return self._perturbed_params.squeeze()
