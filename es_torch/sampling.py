import torch
from jaxtyping import Float
from torch import Tensor

from es_torch.optim import Sampler


def antithetic(npop: int, nparams: int, g: torch.Generator) -> Float[Tensor, "npop nparams"]:
    assert npop % 2 == 0, f"Population size must be even for antithetic sampling, got {npop}"
    eps = torch.randn((npop // 2, nparams), generator=g)
    return torch.cat([eps, -eps], 0)


def normal(npop: int, nparams: int, g: torch.Generator) -> Float[Tensor, "npop nparams"]:
    return torch.randn((npop, nparams), generator=g)


SAMPLERS: dict[str, Sampler] = {
    "antithetic": antithetic,
    "normal": normal,
}

