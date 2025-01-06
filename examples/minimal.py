from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass
class Config:
    npop: int
    lr: float
    std: float
    weight_decay: float
    seed: int
    device: str


type EvalFxn = Callable[[Float[Tensor, "npop params"]], Float[Tensor, "npop"]]


class ES:
    def __init__(self, config: Config, params: Float[Tensor, "params"], eval_fxn: EvalFxn) -> None:
        self.cfg = config
        self.params = params
        self._eval_policies = eval_fxn
        self.g = torch.Generator().manual_seed(config.seed)

    @torch.inference_mode()
    def step(self) -> None:
        noise = torch.cat([eps := torch.randn((self.cfg.npop // 2, len(self.params)), generator=self.g), -eps], 0)
        perturbations = self.cfg.std * noise
        rewards = self._eval_policies(self.params.unsqueeze(0) + perturbations)
        rewards = (rewards.argsort().argsort() - ((len(rewards) - 1) / 2)) / (len(rewards) - 1)
        gradient = self.cfg.lr / (self.cfg.npop * self.cfg.std) * torch.einsum("np,n->p", perturbations, rewards)
        self.params += gradient - self.cfg.lr * self.cfg.weight_decay * self.params
