"""Inspect a trained Q-table checkpoint: coverage, learned policy, and how far
it has moved from the TWAP benchmark. Pure numpy -- no GPU, no simulation, and
safe to run against a checkpoint while training is still writing new ones.

    python -m rl_execution.inspect_policy checkpoints/qtable_v2.npz
"""

import argparse

import numpy as np

from rl_execution.execution_agent import ACTION_LEVELS, N_DECISIONS
from rl_execution.qlearning import (N_ACTIONS, N_INVENTORY_BUCKETS, N_STATES, STATE_DIMS,
                                     STATE_FEATURES, QLearningPolicy)

TWAP_ACTION = 1  # rl_execution.qlearning.TWAPPolicy always selects this level


def main(path):
    policy = QLearningPolicy.load(path)
    q = policy.q
    visited_mask = (q != 0).any(axis=1)
    n_visited = int(visited_mask.sum())
    n_entries = int((q != 0).sum())

    print("=" * 78)
    print(f"Q-TABLE: {path}")
    print("=" * 78)
    print(f"episodes trained : {policy.episodes_trained}")
    print(f"epsilon          : {policy.epsilon:.4f}   (exploration floor {policy.epsilon_min})")
    print(f"alpha            : {policy.alpha:.4f}")
    print(f"state features   : {STATE_FEATURES}  dims={STATE_DIMS}")
    print(f"states visited   : {n_visited}/{N_STATES} ({n_visited / N_STATES:.1%})")
    print(f"entries updated  : {n_entries}/{q.size} ({n_entries / q.size:.1%})")

    if n_visited == 0:
        print("\nNo state has been updated yet -- too early to read a policy off this table.")
        return

    # Greedy action per VISITED state only: argmax over an all-zero row is a tie broken
    # toward action 0, which would otherwise fabricate a spurious 'passive' preference.
    greedy = q.argmax(axis=1)[visited_mask]
    counts = np.bincount(greedy, minlength=N_ACTIONS)
    print("\nGREEDY ACTION over visited states")
    for i, c in enumerate(counts):
        name = ACTION_LEVELS[i]["name"]
        marker = "  <- TWAP benchmark's action" if i == TWAP_ACTION else ""
        print(f"  {i} {name:16s} {c:4d}  ({c / n_visited:5.1%}){marker}")

    differs = int((greedy != TWAP_ACTION).sum())
    print(f"\nStates where the learned policy differs from TWAP: {differs}/{n_visited} "
          f"({differs / n_visited:.1%})")
    if differs == 0:
        print("  -> policy is currently indistinguishable from the TWAP benchmark")

    _policy_grid(q, visited_mask)

    spread = q[visited_mask].max(axis=1) - q[visited_mask].min(axis=1)
    print(f"\nQ-value spread within a visited state (max-min): "
          f"median={np.median(spread):.2f}  max={spread.max():.2f}")
    print("  A spread near zero means actions are barely distinguished yet -- the usual")
    print("  signature of too few visits per entry rather than of a genuine tie.")


def _policy_grid(q, visited_mask):
    """The state space is (time_remaining x inventory_remaining), so the policy can be
    printed as a grid -- the most direct way to see whether it learned a schedule."""
    if STATE_FEATURES != ("time", "inventory"):
        return
    print("\nPOLICY GRID  (rows = time remaining, cols = inventory remaining; '.' = unvisited)")
    print("             " + "".join(f"{c:>6}" for c in
                                     [f"{i / N_INVENTORY_BUCKETS:.0%}" for i in range(N_INVENTORY_BUCKETS)]))
    for t in range(N_DECISIONS + 1):
        cells = []
        for inv in range(N_INVENTORY_BUCKETS):
            s = int(np.ravel_multi_index((t, inv), STATE_DIMS))
            cells.append(f"{q[s].argmax():>6}" if visited_mask[s] else f"{'.':>6}")
        print(f"  t_rem {t:2d}/{N_DECISIONS} " + "".join(cells))
    print("  (cell = greedy action index; see GREEDY ACTION list above for names)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to a Q-table .npz")
    args = parser.parse_args()
    main(args.checkpoint)
