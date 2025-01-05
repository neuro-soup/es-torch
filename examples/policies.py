from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class SimpleMLPConfig:
    obs_dim: int
    act_dim: int
    hidden_dim: int


class SimpleMLP(nn.Module):
    def __init__(self, config: SimpleMLPConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.act_dim),
        )

    def init_weights(self) -> None:
        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(_init_weights)

    def forward(self, obs: Tensor) -> Tensor:
        return self.network(obs)
