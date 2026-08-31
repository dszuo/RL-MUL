"""Deep Q-learning over compressor-tree edits.

Standard DQN with a target network and a replay buffer, plus two things the
design space forces:

* **Action masking.**  Only a handful of the ``4 * (2W-1)`` edits are legal in
  any tree.  Illegal actions are pushed to ``-inf`` before both the argmax used
  to act and the max used to bootstrap, so the target never backs up a value the
  agent could not have taken.  That means the mask of the *next* state has to be
  stored in the buffer alongside the transition.
* **Double DQN.**  Bootstrapping through a max over the masked actions
  overestimates badly; selecting with the online network and evaluating with the
  target costs one extra forward pass and removes most of the bias.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import QNetwork


@dataclass
class DQNConfig:
    lr: float = 1e-2
    gamma: float = 0.08
    batch_size: int = 64
    buffer_size: int = 10_000
    learning_starts: int = 64
    target_update: int = 250
    train_frequency: int = 1
    eps_start: float = 0.9
    eps_end: float = 0.10
    eps_decay_steps: int = 500
    grad_clip: float = 1.0
    # Q-network trunk: one entry per stage, giving its block count and channel
    # width. The default is ResNet-18. Shrink it to trade capacity for speed --
    # the observation is tiny, so a smaller trunk is often enough.
    blocks: tuple[int, ...] = (2, 2, 2, 2)
    channels: tuple[int, ...] = (64, 128, 256, 512)


class ReplayBuffer:
    """Fixed-capacity transition store, preallocated to avoid per-step churn."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], n_actions: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.size = 0
        self.pos = 0

    def push(self, obs, action, reward, next_obs, next_mask) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_mask[i] = next_mask
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: random.Random):
        idx = np.fromiter(
            (rng.randrange(self.size) for _ in range(batch_size)),
            dtype=np.int64, count=batch_size,
        )
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.next_mask[idx])

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int,
                 config: DQNConfig | None = None, device: str = "cpu", seed: int = 0):
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.rng = random.Random(seed)

        def make() -> QNetwork:
            return QNetwork(
                n_actions,
                blocks=tuple(self.config.blocks),
                widths=tuple(self.config.channels),
                in_channels=obs_shape[0],
            ).to(self.device)

        self.online = make()
        self.target = make()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.RMSprop(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_size, obs_shape, n_actions)
        self.steps = 0

    @property
    def epsilon(self) -> float:
        c = self.config
        decay = math.exp(-self.steps / max(1, c.eps_decay_steps))
        return c.eps_end + (c.eps_start - c.eps_end) * decay

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Greedy actions for a batch of observations, one forward pass.

        A row with no legal action would come back as an arbitrary index once
        the all -inf row is passed to argmax, which is worse than stopping: the
        caller would go on to apply an illegal edit.
        """
        empty = np.flatnonzero(~np.asarray(mask).any(axis=1))
        if empty.size:
            raise ValueError(f"no legal action available for observation(s) {empty.tolist()}")
        self.online.eval()
        q = self.online(torch.as_tensor(obs, device=self.device))
        q = q.masked_fill(~torch.as_tensor(mask, device=self.device), -torch.inf)
        self.online.train()
        return q.argmax(dim=1).cpu().numpy()

    def observe(self, obs, action, reward, next_obs, next_mask) -> None:
        self.buffer.push(obs, action, reward, next_obs, next_mask)
        self.steps += 1

    def update(self) -> float | None:
        """One gradient step; ``None`` while the buffer is still filling."""
        c = self.config
        if len(self.buffer) < max(c.batch_size, c.learning_starts):
            return None
        if self.steps % c.train_frequency != 0:
            return None

        obs, actions, rewards, next_obs, next_mask = self.buffer.sample(c.batch_size, self.rng)
        obs = torch.as_tensor(obs, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_mask = torch.as_tensor(next_mask, device=self.device)

        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: pick the next action with the online network, value
            # it with the target network.
            target_q = self.target(next_obs).masked_fill(~next_mask, -torch.inf)
            online_next = self.online(next_obs).masked_fill(~next_mask, -torch.inf)
            best = online_next.argmax(dim=1, keepdim=True)
            next_value = target_q.gather(1, best).squeeze(1)
            # A state with no legal successor contributes no bootstrap term.
            next_value = torch.nan_to_num(next_value, neginf=0.0)
            expected = rewards + c.gamma * next_value

        loss = F.smooth_l1_loss(q, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_value_(self.online.parameters(), c.grad_clip)
        self.optimizer.step()

        if self.steps % c.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = int(state.get("steps", 0))
