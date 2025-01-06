# es-torch

<div align="center">
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch" />
    </a>
    <a href="https://github.com/neuro-soup/evochi">
        <img src="https://img.shields.io/badge/Distributed-Evochi-00ADD8?style=for-the-badge" alt="Evochi" />
    </a>
</div>
<br/>

A lightweight PyTorch implementation of Evolution Strategies (ES) based on the paper [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864).

### 🌟 Key Features

- Minimal Core Implementation: Full ES algorithm in just a few lines of code
- Distributed Training: Scale across multiple machines using Evochi
- Modular design with support for different sampling strategies and reward transformations

### 💡 Core Algorithm

```python
noise = torch.cat([eps := torch.randn((self.cfg.npop // 2, len(self.params)), generator=self.g), -eps], 0) # antithetic sampling
perturbations = self.cfg.std * noise # explore in parameter space
rewards = self._eval_policies(self.params.unsqueeze(0) + perturbations)  # evaluate perturbed policies
rewards = (rewards.argsort().argsort() - ((len(rewards) - 1) / 2)) / (len(rewards) - 1) # centered rank transformation
gradient = self.cfg.lr / (self.cfg.npop * self.cfg.std) * torch.einsum("np,n->p", perturbations, rewards)
self.params += gradient - self.cfg.lr * self.cfg.weight_decay * self.params # gradient ascent
```

### 📦 Implementation Overview

- **Minimal Version**: A complete, minimalistic centralized version of the core algorithm can be found at [examples/minimal.py](https://github.com/neuro-soup/es-torch/blob/main/examples/minimal.py) (37 lines), with a usage example at [examples/train_half_cheetah_minimal.py](https://github.com/neuro-soup/es-torch/blob/main/examples/train_half_cheetah_minimal.py) (136 lines)
- **Full optimizer**: [es_torch/optim.py](https://github.com/neuro-soup/es-torch/blob/main/es_torch/optim.py) supports distributed training, a distributed training example using [evochi](https://github.com/neuro-soup/evochi/tree/master) as a minimal server for sharing rewards can be found at [examples/train_half_cheetah_dist.py](https://github.com/neuro-soup/es-torch/blob/main/examples/train_half_cheetah_dist.py).

### Usage

To run the examples, first install the requirements:
```bash
pip install -r requirements.txt
```

The local examples can be run out of the box with the default config, e.g.:
```bash
python examples/train_half_cheetah_minimal.py
```

The distributed example requires launching an [evochi](https://github.com/neuro-soup/evochi/tree/master) server:
```bash
EVOCHI_JWT_SECRET="secret" EVOCHI_POPULATION_SIZE=1440 go run github.com/neuro-soup/evochi/cmd/evochi@latest
```
Note that the population size configured on the server needs to match the population size in the training script.

Then execute the distributed training script on each worker, e.g.:
```bash
python examples/train_half_cheetah_dist.py --wandb --name myrun --server localhost:8080 --bs 50
```
The batch size should be set according to the workers' resources (to run that many environments in parallel).
Per default, it will use the number of logical CPUs available on the machine.
Note that the `--wandb` flag should be used only on one of the workers.
Get a list of available arguments by adding the `--help` flag.

> [!TIP]
The distributed training script supports logging and resuming runs with wand, and checkpointing, per worker.
A run will continue as long as at least 1 worker is running, so even if the worker that's logging to wandb is stopped, you can resume it from another worker (and no information is lost, as each worker has the same information).

If checkpoints were saved during training via `--ckpt`, you can render videos of a all saved checkpoints with:
```bash
python examples/render.py all "HalfCheetah-v5"
```
or supply a specific checkpoint path to render, instead of `all`.

### 📊 Results

We trained on the [HalfCheetah-v5 gym](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) for 1000 epochs with varying population sizes, and achieved average rewards of up to 4000.
A detailed report on the training can be found here: [Wandb Report](https://wandb.ai/maxw/ES-HalfCheetah/reports/ES-HalfCheetah--VmlldzoxMDgyNTA5MQ?accessToken=mx2jsa0zjqoew8iznpjdqgvnge63l4voc9n2493dx3zxld9yvjt3p59x5n6ijqhf)

We also conducted a sweep over over hyperparameters using wandb (see the report).
To run your own hyperparameter sweep:
```bash
python examples/sweep_half_cheetah.py
```
