from __future__ import annotations, annotations

import argparse
import asyncio
import multiprocessing
import pickle
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional

import grpc
import gymnasium as gym
import numpy as np
import torch
import wandb
from google.protobuf import timestamp_pb2
from gymnasium import VectorizeMode

from es_torch.distributed import distributed_pb2 as proto, distributed_pb2_grpc as services
from es_torch.distributed_optim import Config as ESConfig, ES
from examples.policies import SimpleMLP, SimpleMLPConfig
from examples.utils import (
    ESArgumentHandler,
    ExperimentConfig,
    WandbArgumentHandler,
    WandbConfig,
    reshape_params,
    short_uuid,
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


@dataclass
class WorkerState:
    """Used for transferring state to newly joined workers."""

    epoch: int
    optim_params: torch.Tensor
    optim_rng_state: torch.ByteTensor
    wandb_run_id: str | None = None


class Worker:
    def __init__(self, config: Config, server_address: str) -> None:
        self.config = config

        self.channel = grpc.aio.insecure_channel(server_address)
        self.stub = services.ESServiceStub(self.channel)
        self.worker_id: int | None = None
        self.done: bool = False

        self.optim: ES | None = None
        self.epoch: int = 0
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None
        self.wandb_run_id: str | None = None

    async def run(self) -> None:
        heartbeat_task = asyncio.create_task(self._send_heartbeats())
        subscribe_task = asyncio.create_task(self._handle_server_events())
        try:
            await asyncio.gather(heartbeat_task, subscribe_task)
        except Exception as e:
            print(f"Worker error: {e}")
            raise
        finally:
            await self.channel.close()

    async def _send_heartbeats(self) -> None:
        while not self.done:
            if self.worker_id is None:
                print("Waiting for worker ID...")
                await asyncio.sleep(1)
                continue
            try:
                timestamp = timestamp_pb2.Timestamp()
                timestamp.FromDatetime(datetime.now())
                await self.stub.Heartbeat(proto.HeartbeatRequest(id=self.worker_id, timestamp=timestamp))
                await asyncio.sleep(10)
            except Exception as e:
                print(f"Heartbeat error: {e} on worker {self.worker_id}")
                await asyncio.sleep(1)

    async def _handle_server_events(self) -> None:
        try:
            response_fxns = {
                proto.ServerEventType.HELLO: self._handle_hello,
                proto.ServerEventType.EVALUATE_BATCH: self._handle_evaluate_batch,
                proto.ServerEventType.OPTIM_STEP: self._handle_optim_step,
                proto.ServerEventType.SEND_STATE: self._handle_send_state,
            }
            responses = self.stub.Subscribe(
                proto.SubscribeRequest(
                    num_cpus=multiprocessing.cpu_count(),
                    num_pop=self.config.es.n_pop,
                    device=self.config.device,
                )
            )
            async for res in responses:
                print(f"Received {res.type} event")
                if self.epoch >= self.config.epochs:
                    self.done = True
                    if self.wandb_run:
                        self.wandb_run.finish()
                    break
                await response_fxns[res.type](getattr(res, res.WhichOneof("event")))  # noqa # nvm the getattr stuff

        except Exception as e:
            print(f"Subscription error: {e}")
            traceback.print_exc()
            await asyncio.sleep(1)

    async def _handle_hello(self, res: proto.ServerEventType.HelloEvent) -> None:
        print("Hello has been called")
        self.worker_id = res.id
        if not res.init_state:
            initial_params = torch.nn.utils.parameters_to_vector(SimpleMLP(self.config.policy).parameters())
            self.optim = ES(self.config.es, params=initial_params, device=self.config.device)
            self.wandb_run_id = short_uuid() if self.config.wandb.enabled else None
            self.wandb_run = (
                wandb.init(
                    id=self.wandb_run_id,
                    project=self.config.wandb.project,
                    name=self.config.wandb.name,
                    tags=self.config.wandb.tags,
                    entity=self.config.wandb.entity,
                    config=vars(self.config),
                )
                if self.config.wandb.enabled
                else None
            )
        else:
            worker_state: WorkerState = pickle.loads(res.init_state)
            rng_state = worker_state.optim_rng_state  # .to(self.config.device)
            print(rng_state)
            self.optim = ES(
                self.config.es, params=worker_state.optim_params, device=self.config.device, rng_state=rng_state
            )
            self.epoch = worker_state.epoch
            self.wandb_run_id = worker_state.wandb_run_id

    async def _handle_evaluate_batch(self, res: proto.ServerEventType.EvaluateBatchEvent) -> None:
        perturbed_params = self.optim.get_perturbed_params()
        policy_batch_slice = slice(res.pop_slice.start, res.pop_slice.end)
        rewards = self._evaluate_policy_batch(perturbed_params[policy_batch_slice, :])
        print(
            f"(Worker {self.worker_id}): Epoch {self.epoch} | Slice [{res.pop_slice.start}:{res.pop_slice.end}] | Mean reward: {rewards.mean()} | Max reward: {rewards.max()}"
        )
        await self.stub.Done(
            proto.DoneRequest(
                id=self.worker_id,
                slice=res.pop_slice,
                batch_rewards=[r.cpu().numpy().astype(np.float32).tobytes() for r in rewards],
            )
        )

    async def _handle_optim_step(self, res: proto.ServerEventType.OptimStepEvent) -> None:
        rewards = torch.tensor(
            [torch.frombuffer(r, dtype=torch.float32) for r in res.rewards], device=self.config.device
        )
        self.optim.step(rewards)
        self.epoch += 1
        mean_reward, max_reward = rewards.mean(), rewards.max()
        print(f"Epoch {self.epoch}/{self.config.epochs}: Mean reward: {mean_reward} | Max reward: {max_reward}")
        print(res.logging)
        print(self.wandb_run_id)
        if self._is_logger(res):
            print("Logging to wandb...")
            self.wandb_run.log(
                {
                    "epoch": self.epoch,
                    "mean_reward": mean_reward,
                    "max_reward": max_reward,
                }
            )
        if self.config.ckpt_every > 0 and self.epoch % self.config.ckpt_every == 0:
            model = SimpleMLP(self.config.policy)
            torch.nn.utils.vector_to_parameters(self.optim.params, model.parameters())
            fp = Path(f"{self.config.ckpt_path}_epoch_{self.epoch}")
            model.save(fp)
            print(f"Saved checkpoint to {fp}")

    async def _handle_send_state(self, res: proto.ServerEventType.SendStateEvent) -> None:
        # a new worker joins and needs the current state
        worker_state = WorkerState(
            epoch=self.epoch,
            optim_params=self.optim.params.to(res.device),
            optim_rng_state=self.optim.generator.get_state(),
            wandb_run_id=self.wandb_run.id,
        )
        await self.stub.SendState(proto.SendStateRequest(id=self.worker_id, state=pickle.dumps(worker_state)))

    def _evaluate_policy_batch(self, policy_params_batch: torch.Tensor) -> torch.Tensor:
        npop = policy_params_batch.size(0)
        env = gym.make_vec("Humanoid-v5", num_envs=npop, vectorization_mode=VectorizeMode.ASYNC)

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

    def _is_logger(self, res: proto.ServerEventType.OPTIM_STEP) -> bool:
        """Check if this worker is the current dedicated logging worker and setup wandb it was not already."""
        is_logger = res.logging and self.wandb_run_id is not None
        if is_logger and self.wandb_run is None:
            self.wandb_run = wandb.init(
                id=self.wandb_run_id,
                resume="must",
                project=self.config.wandb.project,
                name=self.config.wandb.name,
                tags=self.config.wandb.tags,
                entity=self.config.wandb.entity,
                config=vars(self.config),
            )
        return is_logger


def train(config: Config, server: str) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    worker = Worker(config, server_address=server)
    try:
        loop.run_until_complete(worker.run())
    except KeyboardInterrupt:
        print("Training interrupted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_episode_steps", type=int, help="Max steps per episode")
    parser.add_argument("--hid", type=int, help="Hidden layer size")
    parser.add_argument("--ckpt", type=int, help="Save every N epochs. N<=0 disables saving")
    parser.add_argument("--server", type=str, help="IP address of the server", default="localhost:8080")
    ESArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = vars(parser.parse_args())
    cfg = Config.default()
    ESArgumentHandler.update_config(args, cfg)
    WandbArgumentHandler.update_config(args, cfg)
    cfg.epochs = args["epochs"] or cfg.epochs
    cfg.max_episode_steps = args["max_episode_steps"] or cfg.max_episode_steps
    cfg.policy.hidden_dim = args["hid"] or cfg.policy.hidden_dim

    # filename = "humanoid.pt" if not cfg.wandb.enabled else f"humanoid_{run.name}.pt"
    # cfg.ckpt_path = Paths.CKPTS / filename
    # cfg.ckpt_every = args["ckpt"] or cfg.ckpt_every
    #
    pprint(cfg)

    # final_params = train(cfg)
    train(cfg, server=args["server"])

    # model = SimpleMLP(cfg.policy)
    # torch.nn.utils.vector_to_parameters(final_params, model.parameters())
    # fp = cfg.ckpt_path.with_stem(cfg.ckpt_path.stem + "_final")
    # save_policy(
    #     model=model,
    #     model_config=cfg.policy,
    #     fp=fp,
    # )
    # print(f"Saved final checkpoint to {fp}")


# def save_final_checkpoint(model, model_config, fp):
#     model = SimpleMLP(model_config)
#     torch.nn.utils.vector_to_parameters(final_params, model.parameters())
#     fp = cfg.ckpt_path.with_stem(cfg.ckpt_path.stem + "_final")
#     save_policy(
#         model=model,
#         model_config=cfg.policy,
#         fp=fp,
#     )
#     print(f"Saved final checkpoint to {fp}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # necessary
    main()
