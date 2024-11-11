from __future__ import annotations, annotations

import asyncio
import multiprocessing
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import grpc
import gymnasium as gym
import numpy as np
import torch
import wandb
from google.protobuf import timestamp_pb2
from gymnasium import VectorizeMode

from es_torch.distrbuted import Config as ESConfig, ES
from es_torch.es import es_pb2, es_pb2_grpc
from examples.policies import SimpleMLP, SimpleMLPConfig
from examples.utils import (
    ExperimentConfig,
    WandbConfig, reshape_params,
)


@dataclass
class Config(ExperimentConfig):
    policy: SimpleMLPConfig
    epochs: int
    max_episode_steps: int
    env_seed: Optional[int]
    device: str
    ckpt_every: int = -1
    ckpt_path: Optional[str | Path] = None

    @classmethod
    def default(cls) -> Config:
        """More or less from: https://github.com/openai/evolution-strategies-starter/blob/master/configurations/humanoid.json"""
        return cls(
            es=ESConfig(
                n_pop=1440,
                lr=0.01,
                std=0.02,
                weight_decay=0.005,
                sampling_strategy="antithetic",
                reward_transform="centered_rank",
                seed=42,
            ),
            wandb=WandbConfig(
                project="ES-Humanoid",
            ),
            policy=SimpleMLPConfig(
                obs_dim=348,
                act_dim=17,
                hidden_dim=256,
            ),
            epochs=1000,
            max_episode_steps=1000,
            env_seed=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


class Worker:
    worker_id: int
    optim: ES

    def __init__(self, config: Config, server_address: str) -> None:
        self.config = config
        self.channel = grpc.insecure_channel(server_address)
        self.stub = es_pb2_grpc.ESServiceStub(self.channel)

    async def run(self) -> None:
        # Register with the server
        num_cpus = multiprocessing.cpu_count()
        resp = self.stub.Hello(es_pb2.HelloRequest(num_cpus=num_cpus))
        self.optim = pickle.loads(resp.state)
        self.worker_id = resp.id

        heartbeat_task = asyncio.create_task(self._send_heartbeats())
        subscribe_task = asyncio.create_task(self._handle_server_events())

        try:
            await asyncio.gather(heartbeat_task, subscribe_task)
        except Exception as e:
            print(f"Worker error: {e}")
            raise

    async def _send_heartbeats(self) -> None:
        while True:
            try:
                timestamp = timestamp_pb2.Timestamp()
                timestamp.FromDatetime(datetime.now())
                self.stub.Heartbeat(es_pb2.HeartbeatRequest(id=self.worker_id, timestamp=timestamp))
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Heartbeat error: {e}")
                await asyncio.sleep(1)

    async def _handle_server_events(self):
        while True:
            try:
                subscribe_request = es_pb2.SubscribeRequest(id=self.worker_id)
                responses = self.stub.Subscribe(subscribe_request)
                for response in responses:
                    if response.type == es_pb2.ServerEventType.SEND_STATE:
                        self.stub.SendState(es_pb2.SendStateRequest(id=self.worker_id, state=pickle.dumps(self.optim)))
                    # elif response.type == es_pb2.ServerEventType.STATE_UPDATE:
                    #     state_tensor = torch.frombuffer(response.updated_state, dtype=torch.float32)
                    #     self.optim.params = state_tensor.reshape(1, -1)
                    elif response.type == es_pb2.ServerEventType.NEXT_EPOCH:
                        rewards = torch.frombuffer(response.rewards, dtype=torch.float32)
                        perturbed_params = self.optim.get_perturbed_params()
                        results = self.evaluate_policy_batch(perturbed_params)

                        # Update optimizer and send results back to server
                        self.optim.step(perturbed_params=perturbed_params, rewards=rewards)
                        self.stub.Done(es_pb2.DoneRequest(id=self.worker_id, reward=results.numpy().tobytes()))

            except Exception as e:
                print(f"Subscription error: {e}")
                await asyncio.sleep(1)

    def evaluate_policy_batch(self, policy_params_batch: torch.Tensor) -> torch.Tensor:
        env = gym.make_vec("Humanoid-v5", num_envs=self.config.es.n_pop, vectorization_mode=VectorizeMode.ASYNC)

        obs, _ = env.reset(seed=self.config.env_seed)
        dones = np.zeros(env.num_envs, dtype=bool)
        total_rewards = np.zeros(env.num_envs, dtype=np.float32)

        base_model = SimpleMLP(self.config.policy).to("meta")
        params_flat = policy_params_batch.to(self.config.device)
        stacked_params_dict = reshape_params(params_flat, base_model)

        def call_model(params: dict[str, torch.Tensor], o: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(base_model, params, (o,))

        vmapped_call_model = torch.vmap(call_model, in_dims=(0, 0))

        for _ in range(self.config.max_episode_steps):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config.device)
            actions = vmapped_call_model(stacked_params_dict, obs)
            obs, rewards, terminations, truncations, _ = env.step(actions.cpu().numpy())
            dones = dones | terminations | truncations
            total_rewards += rewards * ~dones
            if dones.all():
                break

        env.close()
        return torch.tensor(total_rewards, device=self.config.device)


def train(config: Config):
    worker = Worker(config, server_address="localhost:8080")

    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
        )

    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        if config.wandb.enabled:
            wandb.finish()


if __name__ == "__main__":
    config = Config.default()
    train(config)
