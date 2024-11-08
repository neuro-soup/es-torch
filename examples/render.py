import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import torch
from torch import nn

from examples.policies import SimpleMLP
from examples.utils import Paths, load_policy


def parse_render_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
        help="Name of the model checkpoint. If name is 'all', renders an episode for all ckpts based on ckpt names",
    )
    parser.add_argument("env", type=str, help="Environment ID")
    parser.add_argument("--steps", type=int, help="Maximum number of steps to render", default=500)
    return parser.parse_args()


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
        if done or truncated:
            break
    env.close()
    print(f"Episode finished with total reward: {total_reward}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_stem(f"{output_path.stem}_reward_{total_reward:.0f}")
    imageio.mimsave(str(output_path), frames, fps=render_fps)
    print(f"Saved to {output_path}")
    return total_reward


def render(args: argparse.Namespace) -> None:
    if args.name == "all":
        prefix = {"Humanoid-v5": "humanoid", "HalfCheetah-v5": "halfcheetah"}
        for ckpt in Paths.CKPTS.iterdir():
            if prefix[args.env] in ckpt.stem.lower():
                print(f"Rendering {ckpt.stem}")
                policy = load_policy(ckpt_path=ckpt, policy_class=SimpleMLP)
                render_episode(
                    model=policy,
                    env_id=args.env,
                    max_episode_steps=args.steps,
                    output_path=Paths.VIDEOS / f"{ckpt.stem}.mp4",
                )
            else:
                print(f"Skipping {ckpt.stem}")
        return
    ckpt_path = Paths.CKPTS / args.name
    assert ckpt_path.exists(), f"Could not find checkpoint at {ckpt_path}"

    policy = load_policy(ckpt_path=ckpt_path, policy_class=SimpleMLP)

    render_episode(
        model=policy,
        env_id=args.env,
        max_episode_steps=args.steps,
        output_path=Paths.VIDEOS / f"{args.name}.mp4",
    )


def main() -> None:
    args = parse_render_args()
    match args.env:
        case "HalfCheetah-v5":
            render(args)
        case "Humanoid-v5":
            render(args)
        case _:
            raise NotImplementedError(f"Rendering for {args.env} is not implemented")


if __name__ == "__main__":
    main()
