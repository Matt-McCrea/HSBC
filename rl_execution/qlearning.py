"""Tabular Q-learning for the RL execution environment (deliverable 5).
Deliberately NOT deep RL (no neural network policy) -- the spec explicitly
scopes this to tabular/linear function approximation given the time and
compute budget. State discretization matches the buckets
rl_execution.execution_agent.RLExecutionAgent already computes as its
observation.
"""

import json
import os

import numpy as np

from rl_execution.execution_agent import ACTION_LEVELS, N_DECISIONS

N_ACTIONS = len(ACTION_LEVELS)
N_INVENTORY_BUCKETS = 5
N_SPREAD_BUCKETS = 4  # matches execution_agent.SPREAD_BUCKETS -> 4 buckets
N_VOL_BUCKETS = 3      # matches execution_agent.VOL_BUCKETS_TICKS -> 3 buckets
N_OFI_BUCKETS = 3      # matches execution_agent.OFI_BUCKETS -> 3 buckets

STATE_DIMS = (N_DECISIONS + 1, N_INVENTORY_BUCKETS, N_SPREAD_BUCKETS, N_VOL_BUCKETS, N_OFI_BUCKETS)
N_STATES = int(np.prod(STATE_DIMS))


def state_to_index(obs: dict) -> int:
    time_idx = int(round(obs["time_remaining_frac"] * N_DECISIONS))
    time_idx = min(max(time_idx, 0), N_DECISIONS)
    inv_idx = min(int(obs["inventory_remaining_frac"] * N_INVENTORY_BUCKETS), N_INVENTORY_BUCKETS - 1)
    inv_idx = max(inv_idx, 0)
    spread_idx = min(max(int(obs["spread_bucket"]), 0), N_SPREAD_BUCKETS - 1)
    vol_idx = min(max(int(obs["vol_bucket"]), 0), N_VOL_BUCKETS - 1)
    ofi_idx = min(max(int(obs["ofi_bucket"]), 0), N_OFI_BUCKETS - 1)
    return np.ravel_multi_index((time_idx, inv_idx, spread_idx, vol_idx, ofi_idx), STATE_DIMS)


class QLearningPolicy:

    def __init__(self, alpha=0.1, alpha_decay=1.0, alpha_min=0.01,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 gamma=0.99, random_state=None):
        self.q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.rng = random_state or np.random.RandomState()
        self.episodes_trained = 0

    def select_action(self, obs, greedy=False):
        s = state_to_index(obs)
        if not greedy and self.rng.rand() < self.epsilon:
            return int(self.rng.randint(0, N_ACTIONS))
        return int(np.argmax(self.q[s]))

    def update(self, obs, action, reward, next_obs, done):
        s = state_to_index(obs)
        target = reward
        if not done:
            s_next = state_to_index(next_obs)
            target += self.gamma * np.max(self.q[s_next])
        self.q[s, action] += self.alpha * (target - self.q[s, action])

    def end_episode(self):
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez(path, q=self.q, episodes_trained=self.episodes_trained,
                 alpha=self.alpha, epsilon=self.epsilon,
                 hyperparams=json.dumps({
                     "alpha_decay": self.alpha_decay, "alpha_min": self.alpha_min,
                     "epsilon_decay": self.epsilon_decay, "epsilon_min": self.epsilon_min,
                     "gamma": self.gamma,
                 }))

    @classmethod
    def load(cls, path, random_state=None):
        data = np.load(path, allow_pickle=True)
        hp = json.loads(str(data["hyperparams"]))
        policy = cls(alpha=float(data["alpha"]), alpha_decay=hp["alpha_decay"], alpha_min=hp["alpha_min"],
                     epsilon=float(data["epsilon"]), epsilon_decay=hp["epsilon_decay"], epsilon_min=hp["epsilon_min"],
                     gamma=hp["gamma"], random_state=random_state)
        policy.q = data["q"]
        policy.episodes_trained = int(data["episodes_trained"])
        return policy


class TWAPPolicy:
    """Fixed baseline: always the 'light' (index 1) participation level --
    a straight equal-size-per-slice schedule, the standard execution
    benchmark. No state/learning involved."""

    def select_action(self, obs, greedy=True):
        return 1


class ConstantParticipationPolicy:
    """Fixed baseline: always the same action level, configurable."""

    def __init__(self, action_idx=2):
        self.action_idx = action_idx

    def select_action(self, obs, greedy=True):
        return self.action_idx
