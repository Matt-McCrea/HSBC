"""Self-steering session driver for a booked GPU block.

Runs preflight -> calibration -> training -> offline model selection -> evaluation,
with a diagnostic gate between each stage deciding what the next one does. 72 hours
is too long to babysit, and several code paths have only ever been unit-tested, so
the gates exist to catch a broken configuration in the first half hour rather than
after a day of training.

Failure policy is DEGRADE, NEVER IDLE: a failed gate logs loudly, records the reason
in the session state file, and continues with the last configuration proven to work
on GPU. The block always produces something evaluable and the diagnosis is waiting in
the log. Every fallback is recorded, because a patchwork configuration is fine for a
write-up provided it can be stated exactly.

    python -u -m rl_execution.run_session --hours 72 \
        --ckpt-path data/checkpoints/TRADES/<file>.ckpt
"""

import argparse
import json
import os
import subprocess
import sys
import time

from rl_execution.logging_utils import read_episodes

STATE_PATH = "logs/session_state.json"


class Session:
    def __init__(self, args):
        self.args = args
        self.state = {"started": time.time(), "stages": {}, "config": {}, "fallbacks": []}
        self.t0 = time.perf_counter()

    # ---------- bookkeeping ----------

    def hours_left(self):
        return self.args.hours - (time.perf_counter() - self.t0) / 3600.0

    def save(self):
        os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def record(self, stage, **kw):
        self.state["stages"][stage] = {"at_hours_elapsed": round(self.args.hours - self.hours_left(), 2), **kw}
        self.save()

    def fallback(self, stage, reason, using):
        banner = f"[session] FALLBACK in {stage}: {reason} -> using {using}"
        print("\n" + "!" * len(banner), flush=True)
        print(banner, flush=True)
        print("!" * len(banner) + "\n", flush=True)
        self.state["fallbacks"].append({"stage": stage, "reason": reason, "using": using})
        self.save()

    def banner(self, text):
        print("\n" + "=" * 78, flush=True)
        print(f"[session] {text}   ({self.hours_left():.1f}h of budget left)", flush=True)
        print("=" * 78 + "\n", flush=True)

    # ---------- process runner ----------

    def run(self, module, cli_args, stage):
        cmd = [sys.executable, "-u", "-m", f"rl_execution.{module}"] + [str(a) for a in cli_args]
        print(f"[session] $ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        ok = result.returncode == 0
        if not ok:
            print(f"[session] {module} exited {result.returncode}", flush=True)
        self.record(stage, module=module, args=cli_args, returncode=result.returncode)
        return ok

    # ---------- stages ----------

    def stage_preflight(self):
        """Three episodes with the target configuration, checking invariants that only
        hold once the real kernel is involved."""
        self.banner("STAGE 0 - preflight")
        cfg = {"reward_mode": "per-step", "reward_benchmark": "arrival",
               "inventory_penalty": 0.0, "alpha_mode": "visit-count"}
        log = "logs/preflight.jsonl"
        ok = self.run("benchmark", [
            "--n-episodes", 3, "--out", log, "--run-name", "preflight",
            "--reward-mode", cfg["reward_mode"], "--reward-benchmark", cfg["reward_benchmark"],
            "--ckpt-path", self.args.ckpt_path, "--depth-noise", self.args.depth_noise,
        ], "preflight_run")

        from rl_execution.preflight import run_checks
        records = read_episodes(log) if os.path.exists(log) else []
        passed, report = run_checks(records, cfg["reward_mode"], cfg["reward_benchmark"])
        print("\n".join(report), flush=True)

        if not (ok and passed):
            # Match the fallback to the FAILURE. The first version of this switched to
            # terminal rewards for any preflight failure, which was useless the one time
            # it fired: the real fault was unsold inventory (the aggressor never being told
            # its own order filled), and reward mode has no bearing on that. A fallback
            # that does not address the fault is worse than none, because it looks handled.
            text = " ".join(report).lower()
            attribution_fault = "attribution" in text
            environment_fault = ("liquidation incomplete" in text or "over-execution" in text)

            if attribution_fault and not environment_fault:
                self.fallback("preflight", "per-step reward attribution failed",
                               "terminal reward mode (the _v2 configuration proven on GPU)")
                cfg["reward_mode"] = "terminal"
            elif environment_fault:
                # No configuration fixes this; it is a defect in how orders or fills are
                # handled. Say so plainly rather than quietly training on broken episodes.
                self.fallback("preflight",
                               "ENVIRONMENT FAULT: inventory is not being liquidated / fills are "
                               "not being accounted. No reward configuration fixes this -- the "
                               "session will continue so the booking is not wasted, but treat "
                               "every downstream number as suspect until this is resolved",
                               "unchanged configuration, results flagged untrustworthy")
                self.state["results_trustworthy"] = False
            else:
                self.fallback("preflight", "invariant checks failed (conditioning out of range "
                               "or run error)", "unchanged configuration, results flagged")
                self.state["results_trustworthy"] = False
        self.state["config"].update(cfg)
        self.record("preflight", passed=bool(ok and passed), n_failures=len(report) - 1)
        self.save()
        return cfg

    def stage_calibrate(self, cfg):
        """TWAP episodes to measure sigma, the impact coefficient eta, and drift, then
        derive the inventory penalty and the AC baseline's kappa."""
        self.banner("STAGE 1 - calibration")
        log = "logs/calibration.jsonl"
        self.run("benchmark", [
            "--n-episodes", self.args.calibration_episodes, "--out", log,
            "--run-name", "calibration",
            "--reward-mode", cfg["reward_mode"], "--reward-benchmark", cfg["reward_benchmark"],
            "--ckpt-path", self.args.ckpt_path, "--depth-noise", self.args.depth_noise,
        ], "calibration_run")

        from rl_execution.calibrate import calibrate, format_report
        records = read_episodes(log) if os.path.exists(log) else []
        result = calibrate(records, self.args.target_kappa_t)
        print(format_report(result), flush=True)
        for w in result["warnings"]:
            if not result["calibrated"]:
                self.fallback("calibration", w, f"lam={result['lam']}, kappa={result['kappas']}")
        cfg["inventory_penalty"] = result["lam"]
        cfg["kappas"] = result["kappas"]
        self.state["config"].update(cfg)
        self.record("calibration", **{k: v for k, v in result.items() if k != "warnings"})
        return cfg

    def stage_train(self, cfg):
        self.banner("STAGE 2 - training")
        # Reserve time for evaluation; never let training eat the deliverable.
        budget = max(1.0, min(self.args.train_hours, self.hours_left() - self.args.reserve_hours))
        ok = self.run("train", [
            "--n-episodes", 1000, "--max-hours", round(budget, 2), "--side", "SELL",
            "--reward-mode", cfg["reward_mode"], "--reward-benchmark", cfg["reward_benchmark"],
            "--alpha-mode", cfg["alpha_mode"], "--epsilon-decay", self.args.epsilon_decay,
            "--inventory-penalty", cfg["inventory_penalty"],
            "--ckpt-path", self.args.ckpt_path, "--depth-noise", self.args.depth_noise,
            "--checkpoint", "checkpoints/qtable_session.npz", "--out", "logs/train_session.jsonl",
        ], "training")
        self.record("training", ok=ok, budget_hours=round(budget, 2))
        return ok

    def stage_select(self, cfg):
        """Refit the logged trajectories across variants and pick by stability, so the
        policy that gets evaluated is chosen on evidence rather than by default."""
        self.banner("STAGE 3 - offline model selection (no GPU)")
        from rl_execution.refit_qtable import refit, stability
        from rl_execution.qlearning import QLearningPolicy

        log = "logs/train_session.jsonl"
        records = [r for r in (read_episodes(log) if os.path.exists(log) else []) if r.get("trajectory")]
        live = "checkpoints/qtable_session.npz"
        if len(records) < 20:
            self.fallback("selection", f"only {len(records)} refittable episodes",
                           "the live-trained Q-table, flagged unconverged")
            self.record("selection", chosen="live", refittable=len(records))
            return live, {"note": "live table; too few episodes to refit"}

        lam = cfg.get("inventory_penalty") or 0.0
        candidates = {
            "visit_lam": dict(alpha_mode="visit-count", inventory_lambda=lam),
            "visit_lam0": dict(alpha_mode="visit-count", inventory_lambda=0.0),
            "visit_lam2": dict(alpha_mode="visit-count", inventory_lambda=lam * 2),
        }
        scored = {}
        for name, kw in candidates.items():
            rows = stability(records, **kw)
            churn = (rows[-1]["changed"][0] / rows[-1]["changed"][1]) if rows and rows[-1]["changed"] and rows[-1]["changed"][1] else 1.0
            q, visits, *_ = refit(records, **kw)
            scored[name] = {"final_churn": round(churn, 4),
                             "states_visited": int((visits > 0).any(axis=1).sum())}
            print(f"  {name:12s} churn={churn:.1%}  states={scored[name]['states_visited']}", flush=True)

        best = min(scored, key=lambda n: scored[n]["final_churn"])
        q, visits, *_ = refit(records, **candidates[best])
        policy = QLearningPolicy(alpha_mode="visit-count")
        policy.q, policy.visits = q, visits
        policy.episodes_trained = len(records)
        policy.epsilon = policy.epsilon_min
        out = "checkpoints/qtable_selected.npz"
        policy.save(out)
        print(f"\n  selected: {best} (lowest greedy churn) -> {out}", flush=True)
        self.record("selection", chosen=best, scores=scored, out=out)
        return out, scored

    def stage_evaluate(self, cfg, qtable):
        self.banner("STAGE 4 - evaluation")
        from rl_execution.evaluate import run_comparison
        from rl_execution.qlearning import QLearningPolicy, TWAPPolicy
        from rl_execution.baselines import ACSchedulePolicy

        policies = {"twap": TWAPPolicy()}
        for kappa in cfg.get("kappas") or []:
            policies[f"ac_k{kappa:g}"] = ACSchedulePolicy(kappa=kappa)
        if qtable and os.path.exists(qtable):
            policies["qlearning"] = QLearningPolicy.load(qtable)

        # Split what remains between the policy comparison and the sampler comparison,
        # keeping the sampler result (the spec's most important one) protected.
        remaining = max(1.0, self.hours_left() - 0.5)
        sampler_hours = min(self.args.sampler_hours, remaining * 0.5)
        policy_hours = max(0.5, remaining - sampler_hours)

        print(f"[session] policy comparison ~{policy_hours:.1f}h, sampler ~{sampler_hours:.1f}h", flush=True)
        pol = run_comparison(
            symbol=self.args.symbol, n_seeds=self.args.eval_seeds, policies=policies,
            ckpt_path=self.args.ckpt_path, skip_ddpm=True,
            max_hours_per_arm=policy_hours / max(1, len(policies)),
            out_path="logs/eval_policies.jsonl")
        self.record("eval_policies", results={k: v for k, v in pol.items()})

        samp = run_comparison(
            symbol=self.args.symbol, n_seeds=self.args.eval_seeds,
            policies={"twap": TWAPPolicy()}, ckpt_path=self.args.ckpt_path,
            max_hours_per_arm=sampler_hours / 2.0, out_path="logs/eval_sampler.jsonl")
        self.record("eval_sampler", results={k: v for k, v in samp.items()})
        return pol, samp

    # ---------- driver ----------

    def go(self):
        cfg = self.stage_preflight()
        cfg = self.stage_calibrate(cfg)
        self.stage_train(cfg)
        qtable, _ = self.stage_select(cfg)
        self.stage_evaluate(cfg, qtable)

        self.banner("SESSION COMPLETE")
        print(json.dumps(self.state, indent=2, default=str), flush=True)
        if self.state["fallbacks"]:
            print("\n[session] NOTE: fallbacks were used -- the configuration that produced "
                  "these numbers is not the intended one. See logs/session_state.json.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=72.0, help="total booking length")
    p.add_argument("--train-hours", type=float, default=26.0)
    p.add_argument("--reserve-hours", type=float, default=20.0,
                   help="wall-clock held back from training for evaluation")
    p.add_argument("--sampler-hours", type=float, default=10.0)
    p.add_argument("--calibration-episodes", type=int, default=12)
    p.add_argument("--eval-seeds", type=int, default=20)
    p.add_argument("--epsilon-decay", type=float, default=0.99)
    p.add_argument("--target-kappa-t", type=float, default=2.0)
    p.add_argument("--depth-noise", type=float, default=0.3)
    p.add_argument("--symbol", default="INTC")
    p.add_argument("--ckpt-path", required=True)
    Session(p.parse_args()).go()
