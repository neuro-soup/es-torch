import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import torch
from jaxtyping import Float
from torch import Tensor, nn

ROOT = Path(__file__).parent
VIDEOS = ROOT / "videos"
CKPTS = ROOT / "checkpoints"
DATA = ROOT / "data"
LOGS = ROOT / "logs"


# TODO add options in the help
def add_es_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--npop", type=int, help="Population size")
    parser.add_argument("--std", type=float, help="Standard deviation of noise")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--wd", type=float, help="Weight decay")
    parser.add_argument("--noise", type=str, help="Noise sampling strategy")
    parser.add_argument("--reward", type=str, help="Reward normalization strategy")
    parser.add_argument("--seed", type=int, help="Seed for noise sampling")
    parser.add_argument("--wandb", action="store_true", help="Use wandb")
    parser.add_argument("--name", type=str, help="wandb run name")


def reshape_params(
        params_flat: Float[Tensor, "npop params"], model: nn.Module
) -> dict[str, Float[Tensor, "npop *_"]]:
    npop, _ = params_flat.shape
    param_dict = {}
    param_names = []
    param_shapes = []
    param_sizes = []
    for name, param in model.named_parameters():
        param_names.append(name)
        param_shapes.append(param.shape)
        param_sizes.append(param.numel())
    params_split = torch.split(params_flat, param_sizes, dim=1)
    for name, shape, param in zip(param_names, param_shapes, params_split):
        param_dict[name] = param.view(npop, *shape)
    return param_dict


@torch.inference_mode()
def render_episode(
        model: nn.Module,
        env_id: str,
        output_path: Path,
        max_episode_steps: int = 1000,
        render_fps: int = 30,
        device: str = "cpu",
        **env_kwargs: Any,
) -> float:
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    model = model.to(device).eval()
    frames = []
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(max_episode_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = model(obs_tensor).squeeze(0).cpu().numpy()
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())
        if done or truncated: break
    print(f"Episode finished with total reward: {total_reward}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=render_fps)
    env.close()
    return total_reward


def save_policy(
        model: nn.Module,
        config: Any,
        save_path: str | Path,
) -> None:
    """Save a policy network with its configuration object to a checkpoint."""
    state = {
        'state_dict': model.state_dict(),
        'config': vars(config),
    }
    torch.save(state, save_path)


def load_policy(
        ckpt_path: str | Path,
        policy_class: type[nn.Module],
        config_class: type[Any],
        **kwargs
) -> nn.Module:
    """Load a policy network from a checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file
        policy_class: The class of policy to instantiate
        config_class: The configuration class to use
        **kwargs: Additional arguments passed to the policy constructor
    """
    checkpoint = torch.load(ckpt_path)
    config = config_class(**checkpoint['config'])
    policy = policy_class(config, **kwargs)
    policy.load_state_dict(checkpoint['state_dict'])
    return policy
