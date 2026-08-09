"""Literature baselines for the execution agent to be measured against.

The point of these is that TWAP is not an arbitrary yardstick: under Almgren-Chriss
(2000) with linear temporary impact, the RISK-NEUTRAL optimum is to trade at a
uniform rate, i.e. TWAP is exactly the lambda = 0 member of the AC family. Raising
risk aversion front-loads the trajectory. So TWAP and ACSchedulePolicy are two points
on one curve, and "does the learned policy front-load as risk aversion rises?" becomes
a falsifiable prediction taken from the paper rather than a hope.
"""

import math

from rl_execution.execution_agent import ACTION_LEVELS, N_DECISIONS

# Participation multiple of the Q/N base slice implied by each action, in the order
# the action space defines them. Used to map a desired trade size onto an action.
ACTION_PARTICIPATION = [level["participation"] for level in ACTION_LEVELS]


def ac_target_inventory(time_remaining_frac, kappa):
    """Almgren-Chriss optimal holdings, as a fraction of the parent order.

        x(t)/X = sinh(kappa * (T - t)) / sinh(kappa * T)

    Expressed in the fraction of the window still remaining, u = (T - t)/T, this is
    sinh(kappa_T * u) / sinh(kappa_T) with kappa_T = kappa * T. As kappa_T -> 0 the
    ratio tends to u -- a straight line, i.e. TWAP -- which is why TWAP is the
    risk-neutral member of this family rather than a separate heuristic.
    """
    u = min(max(float(time_remaining_frac), 0.0), 1.0)
    kt = float(kappa)
    if kt <= 1e-9:
        return u  # risk-neutral limit: linear liquidation == TWAP
    return math.sinh(kt * u) / math.sinh(kt)


class ACSchedulePolicy:
    """Trades toward the Almgren-Chriss inventory trajectory.

    `kappa` here is the dimensionless kappa*T (risk aversion over the whole window),
    so kappa=0 is TWAP and larger values front-load harder.

    KNOWN APPROXIMATION, worth stating rather than hiding: AC prescribes a trade
    SIZE, whereas this action space conflates size with aggression (action 1 and
    action 2 are both 1.0x the base slice but differ in whether they cross the
    spread). The schedule therefore picks the action whose participation multiple is
    nearest the size AC wants, breaking ties toward the more passive action, and the
    aggression dimension is left to the RL agent to exploit. The baseline is a
    faithful AC *schedule*, not a claim about optimal order placement.
    """

    def __init__(self, kappa=0.0, n_decisions=N_DECISIONS, prefer_passive=True):
        self.kappa = float(kappa)
        self.n_decisions = n_decisions
        self.prefer_passive = prefer_passive

    def target_after_this_step(self, time_remaining_frac):
        """Inventory AC wants held once this decision's trade is done."""
        step = 1.0 / self.n_decisions
        return ac_target_inventory(max(0.0, time_remaining_frac - step), self.kappa)

    def select_action(self, obs, greedy=True):
        inventory = float(obs["inventory_remaining_frac"])
        target = self.target_after_this_step(obs["time_remaining_frac"])
        desired_fraction = max(0.0, inventory - target)
        # Actions are multiples of the Q/N base slice, so convert the desired fraction
        # of the WHOLE order into that unit before matching.
        desired_multiple = desired_fraction * self.n_decisions

        best_idx, best_gap = 0, None
        for idx, participation in enumerate(ACTION_PARTICIPATION):
            gap = abs(participation - desired_multiple)
            if best_gap is None or gap < best_gap - 1e-12:
                best_idx, best_gap = idx, gap
            elif self.prefer_passive and abs(gap - best_gap) <= 1e-12:
                # Equal-size actions differ only in aggression; AC is silent on that,
                # so take the cheaper (more passive) one rather than paying the spread.
                continue
        return int(best_idx)


def twap_equivalent_kappa():
    """kappa at which ACSchedulePolicy is TWAP. Named so tests and callers can assert
    the equivalence rather than hard-coding a magic zero."""
    return 0.0
