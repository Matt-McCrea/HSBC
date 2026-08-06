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

# Which observation features actually index the Q-table. The environment always COMPUTES
# all five (they're in every observation and every log line), but only these index Q.
#
# WHY only time+inventory by default: the full five-feature space is
# 11*5*4*3*3 = 1980 states x 5 actions = 9900 entries. Measured episode cost is ~460s,
# so a realistic run is ~100-200 episodes = ~1000-2000 TD updates -- most entries would
# never be visited even once, and the policy would come out degenerate for an
# uninteresting reason (state space sized for a budget that doesn't exist) rather than
# an interesting one. time x inventory = 55 states x 5 actions = 275 entries is ~7 visits
# each: thin, but actually learnable, and it is exactly the Almgren-Chriss state
# (time remaining, inventory remaining), so the learned policy is directly comparable to
# the TWAP benchmark. Add features back here when the episode budget supports them.
STATE_FEATURES = ("time", "inventory")

_FEATURE_DIMS = {
    "time": N_DECISIONS + 1,
    "inventory": N_INVENTORY_BUCKETS,
    "spread": N_SPREAD_BUCKETS,
    "vol": N_VOL_BUCKETS,
    "ofi": N_OFI_BUCKETS,
}

STATE_DIMS = tuple(_FEATURE_DIMS[f] for f in STATE_FEATURES)
N_STATES = int(np.prod(STATE_DIMS))


def _feature_index(obs: dict, feature: str) -> int:
    if feature == "time":
        idx = int(round(obs["time_remaining_frac"] * N_DECISIONS))
    elif feature == "inventory":
        idx = int(obs["inventory_remaining_frac"] * N_INVENTORY_BUCKETS)
    else:
        idx = int(obs[f"{feature}_bucket"])
    return min(max(idx, 0), _FEATURE_DIMS[feature] - 1)


def state_to_index(obs: dict) -> int:
    return int(np.ravel_multi_index(
        tuple(_feature_index(obs, f) for f in STATE_FEATURES), STATE_DIMS))


class QLearningPolicy:

    # Defaults sized for the ~100-200 episode budget the measured ~460s/episode allows
    # (see STATE_FEATURES above for the same reasoning applied to the state space):
    #   alpha 0.3   -- with ~7 visits per entry, 0.1 barely moves a value off zero.
    #   eps_decay 0.97 -- reaches the 0.05 floor by ~episode 100. At the old 0.995, epsilon
    #                  is still 0.61 after 100 episodes, i.e. the agent would explore
    #                  randomly for the entire run and never exploit what it learned.
    #   gamma 1.0   -- the reward is a single terminal payment over a fixed 10-step
    #                  horizon, so discounting it has no principled meaning here and
    #                  undiscounted matches the implementation-shortfall definition.
    def __init__(self, alpha=0.3, alpha_decay=1.0, alpha_min=0.01,
                 epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.05,
                 gamma=1.0, random_state=None, alpha_mode="fixed"):
        self.q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        # Per-entry visit counts: needed for alpha_mode="visit-count", and worth carrying
        # regardless -- how often an entry was actually updated is what says whether its
        # value can be trusted, and it is not recoverable from the Q-table alone.
        self.visits = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
        # "visit-count" sets alpha = 1/N(s,a), making Q the running MEAN return. With
        # gamma=1 and a single terminal reward that is exactly what Q is, so averaging is
        # the principled estimator (Robbins-Monro). A fixed alpha=0.3 instead keeps only
        # an effective ~3-episode window: with terminal rewards of sigma ~400 raw units
        # against an action spread of ~140, the estimate tracks noise and the greedy
        # policy visibly churns between checkpoints.
        self.alpha_mode = alpha_mode
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
        self.visits[s, action] += 1
        step_alpha = (1.0 / self.visits[s, action]) if self.alpha_mode == "visit-count" else self.alpha
        self.q[s, action] += step_alpha * (target - self.q[s, action])

    def end_episode(self):
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez(path, q=self.q, visits=self.visits, episodes_trained=self.episodes_trained,
                 alpha=self.alpha, epsilon=self.epsilon,
                 hyperparams=json.dumps({
                     "alpha_decay": self.alpha_decay, "alpha_min": self.alpha_min,
                     "epsilon_decay": self.epsilon_decay, "epsilon_min": self.epsilon_min,
                     "gamma": self.gamma, "alpha_mode": self.alpha_mode,
                 }))

    @classmethod
    def load(cls, path, random_state=None):
        data = np.load(path, allow_pickle=True)
        hp = json.loads(str(data["hyperparams"]))
        policy = cls(alpha=float(data["alpha"]), alpha_decay=hp["alpha_decay"], alpha_min=hp["alpha_min"],
                     epsilon=float(data["epsilon"]), epsilon_decay=hp["epsilon_decay"], epsilon_min=hp["epsilon_min"],
                     gamma=hp["gamma"], random_state=random_state,
                     alpha_mode=hp.get("alpha_mode", "fixed"))
        policy.q = data["q"]
        # visits absent in tables saved before visit-count alpha existed
        if "visits" in data.files:
            policy.visits = data["visits"]
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
