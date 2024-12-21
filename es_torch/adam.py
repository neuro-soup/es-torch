from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from es_torch import optim


@dataclass
class Config(optim.Config):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        assert 0 <= self.beta1 < 1, "Beta1 should be in [0, 1)."
        assert 0 <= self.beta2 < 1, "Beta2 should be in [0, 1)."
        assert 0 <= self.eps, "Epsilon should be non-negative."


class ES(optim.ES):
    cfg: Config  # linters doesn't recognize inherited attributes otherwise...

    def __init__(self, config: Config, params: Float[Tensor, "params"]) -> None:
        super().__init__(config, params)
        self.m = torch.zeros_like(self.params)
        self.v = torch.zeros_like(self.params)
        self.t = 0

    @torch.inference_mode()
    def step(self, rewards: Float[Tensor, "npop"]) -> None:
        rewards = self._transform_reward(rewards)
        grad_estimate = self.cfg.lr / (self.cfg.npop * self.cfg.std) * torch.einsum("np,n->p", self._perturbed_params, rewards)
        self.m = self.cfg.beta1 * self.m + (1 - self.cfg.beta1) * grad_estimate
        self.v = self.cfg.beta2 * self.v + (1 - self.cfg.beta2) * grad_estimate**2
        m_hat = self.m / (1 - self.cfg.beta1 ** (self.t + 1))
        v_hat = self.v / (1 - self.cfg.beta2 ** (self.t + 1))
        self.params += self.cfg.lr * m_hat / (torch.sqrt(v_hat) + self.cfg.eps)
        self.params -= self.cfg.lr * self.cfg.weight_decay * self.params
        self.t += 1
