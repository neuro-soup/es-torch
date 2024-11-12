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
    epoch: int
    optim: ES
    wandb_run: wandb.sdk.wandb_run.Run | None


class Worker:
    def __init__(self, config: Config, server_address: str) -> None:
        self.config = config
        self.channel = grpc.aio.insecure_channel(server_address)
        self.stub = services.ESServiceStub(self.channel)
        self.worker_id: int | None = None
        self.state: WorkerState | None = None
        self._done: bool = False
        self._npop: int | None = None

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
        while not self._done:
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
            self._npop = multiprocessing.cpu_count()
            responses = self.stub.Subscribe(
                proto.SubscribeRequest(
                    num_cpus=self._npop,
                    num_pop=self.config.es.n_pop,
                )
            )
            async for res in responses:
                print(f"Received {res.type} event")
                if self.state and self.state.epoch >= self.config.epochs:
                    self._done = True
                    break
                await response_fxns[res.type](getattr(res, res.WhichOneof("event")))  # noqa # nvm the getattr stuff

            if self.state and self.state.wandb_run:
                self.state.wandb_run.finish()
        except Exception as e:
            print(f"Subscription error: {e}")
            traceback.print_exc()
            await asyncio.sleep(1)

    async def _handle_hello(self, res: proto.ServerEventType.HelloEvent) -> None:
        print("Hello has been called")
        self.worker_id = res.id
        if not res.init_state:
            initial_params = torch.nn.utils.parameters_to_vector(SimpleMLP(self.config.policy).parameters())
            optim = ES(self.config.es, params=initial_params)
            run = (
                wandb.init(
                    project=self.config.wandb.project,
                    name=self.config.wandb.name,
                    tags=self.config.wandb.tags,
                    entity=self.config.wandb.entity,
                    config=vars(self.config),
                )
                if self.config.wandb.enabled
                else None
            )
            self.state = WorkerState(epoch=0, optim=optim, wandb_run=run)
        else:
            self.state = pickle.loads(res.init_state)

    async def _handle_evaluate_batch(self, res: proto.ServerEventType.EvaluateBatchEvent) -> None:
        perturbed_params = self.state.optim.get_perturbed_params()
        policy_batch_slice = slice(res.pop_slice.start, res.pop_slice.end)
        rewards = self._evaluate_policy_batch(perturbed_params[policy_batch_slice, :])
        print(
            f"(Worker {self.worker_id}): Epoch {self.state.epoch} | Slice [{res.pop_slice.start}:{res.pop_slice.end}] | Mean reward: {rewards.max()} | Max reward: {rewards.mean()}"
        )
        await self.stub.Done(
            proto.DoneRequest(
                id=self.worker_id, slice=res.pop_slice, batch_rewards=[r.cpu().numpy().tobytes() for r in rewards]
            )
        )

    async def _handle_optim_step(self, res: proto.ServerEventType.OptimStepEvent) -> None:
        rewards = torch.frombuffer(res.rewards, dtype=torch.float32).flatten().to(self.config.device)
        self.state.optim.step(rewards)
        self.state.epoch += 1
        mean_reward, max_reward = rewards.mean(), rewards.max()
        print(f"Epoch {self.state.epoch} | Mean reward: {mean_reward} | Max reward: {max_reward}")
        if self.state.wandb_run and res.logging:  # one dedicated worker logs to wandb
            self.state.wandb_run.log(
                {
                    "epoch": self.state.epoch,
                    "mean_reward": mean_reward,
                    "max_reward": max_reward,
                }
            )
        if self.config.ckpt_every > 0 and self.state.epoch % self.config.ckpt_every == 0:
            model = SimpleMLP(self.config.policy)
            torch.nn.utils.vector_to_parameters(self.state.optim.params, model.parameters())
            fp = Path(f"{self.config.ckpt_path}_epoch_{self.state.epoch}")
            model.save(fp)
            print(f"Saved checkpoint to {fp}")

    async def _handle_send_state(self, res: proto.ServerEventType.SendStateEvent) -> None:
        # a new worker joins and needs the current state
        await self.stub.SendState(proto.SendStateRequest(id=self.worker_id, state=pickle.dumps(self.state)))

    def _evaluate_policy_batch(self, policy_params_batch: torch.Tensor) -> torch.Tensor:
        env = gym.make_vec("Humanoid-v5", num_envs=self._npop, vectorization_mode=VectorizeMode.ASYNC)

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
