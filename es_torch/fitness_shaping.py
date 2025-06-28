import torch
from jaxtyping import Float
from torch import Tensor

from es_torch.optim import RewardTransform


def centered_rank(rewards: Float[Tensor, "npop"]) -> Float[Tensor, "npop"]:
    """Compute centered ranks with average handling for ties using efficient tensorized operations."""
    sorted_vals, sorted_indices = torch.sort(rewards)
    ranks = torch.empty_like(sorted_indices, dtype=torch.float32)
    ranks[sorted_indices] = torch.arange(len(rewards), dtype=torch.float32, device=rewards.device)

    # Compute average ranks for each group of tied values
    # For each group, the average rank is (first_rank + last_rank) / 2
    # where first_rank = cumsum[i-1] and last_rank = cumsum[i] - 1
    _, inverse_indices, counts = torch.unique_consecutive(sorted_vals, return_inverse=True, return_counts=True)
    cumsum = counts.cumsum(0)
    avg_ranks = cumsum - counts / 2.0 - 0.5  # -0.5 to convert from 1-based to 0-based
    ranks[sorted_indices] = avg_ranks[inverse_indices]  # map average ranks back to original positions

    if len(rewards) <= 1:
        return torch.zeros_like(rewards)

    return (ranks - (len(rewards) - 1) / 2) / (len(rewards) - 1)  # center and normalize ranks to [-0.5, 0.5]


def normalized(rewards: Float[Tensor, "npop"]) -> Float[Tensor, "npop"]:
    """Normalize rewards to have zero mean and unit variance."""
    return (rewards - rewards.mean()) / rewards.std()


TRANSFORMS: dict[str, RewardTransform] = {
    "centered_rank": centered_rank,
    "normalized": normalized,
}
