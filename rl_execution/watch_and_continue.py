"""Wait for the running session's training stage to finish, then take over and run
the frontier evaluation instead of the orchestrator's default one.

Why this exists: run_session proceeds automatically from training into an evaluation
whose wall-clock split was chosen when the sampler comparison was still the headline
result. It now isn't, so that split would spend ~10h on a sampler arm we are
deliberately deprioritising and would evaluate only ONE risk aversion. Catching the
handover by hand means watching a terminal for an arrival time that could be the
middle of the night; this does it instead.

    python -u -m rl_execution.watch_and_continue \
        --ckpt-path data/checkpoints/TRADES/<file>.ckpt 2>&1 | tee logs/phase2.log

Safe to start at any point, including after training has already finished -- it
checks the existing log first and acts immediately if the marker is already there.
Safe to leave running before training completes; it simply waits.
"""

import argparse
import json
import os
import subprocess
import sys
import time

TRAINING_DONE_MARKER = "training complete: checkpoint saved"
SESSION_PATTERN = "rl_execution.run_session"
SESSION_LOG = "logs/session.log"
STATE_PATH = "logs/session_state.json"
TRAIN_LOG = "logs/train_session.jsonl"
LIVE_QTABLE = "checkpoints/qtable_session.npz"


def log(msg):
    print(f"[watch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def training_finished(path=SESSION_LOG):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", errors="replace") as f:
            return TRAINING_DONE_MARKER in f.read()
    except OSError:
        return False


def wait_for_training(poll_seconds, path=SESSION_LOG):
    if training_finished(path):
        log("training marker already present -- proceeding immediately")
        return True
    log(f"waiting for '{TRAINING_DONE_MARKER}' in {path} (polling every {poll_seconds}s)")
    while True:
        time.sleep(poll_seconds)
        if training_finished(path):
            log("training finished")
            return True
        if not _session_running():
            log("session process is gone and no training marker was found -- "
                "stopping rather than guessing. Inspect the log before continuing.")
            return False


def _session_running():
    r = subprocess.run(["pgrep", "-f", SESSION_PATTERN], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def stop_session():
    """Terminate the orchestrator before it starts its own evaluation.

    Its evaluation runs in-process (run_comparison is called directly, not spawned),
    so terminating the parent is sufficient. Any child benchmark/train process from an
    earlier stage has already exited by this point.
    """
    if not _session_running():
        log("session process not running -- nothing to stop")
        return
    log("stopping the orchestrator so its default evaluation does not start")
    subprocess.run(["pkill", "-TERM", "-f", SESSION_PATTERN])
    for _ in range(20):
        time.sleep(1)
        if not _session_running():
            log("orchestrator stopped")
            return
    log("orchestrator did not exit on SIGTERM; sending SIGKILL")
    subprocess.run(["pkill", "-KILL", "-f", SESSION_PATTERN])
    time.sleep(2)


def calibrated_lambda(default=42.0):
    """Read the inventory penalty the calibration stage derived, so the frontier is
    centred on the risk aversion this simulator actually implies."""
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        lam = state.get("stages", {}).get("calibration", {}).get("lam")
        if lam:
            log(f"calibrated lambda from session state: {lam:.3f}")
            return float(lam)
    except (OSError, ValueError, KeyError):
        pass
    log(f"could not read calibrated lambda; falling back to {default}")
    return default


def run(cmd):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd]).returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--poll", type=int, default=30, help="seconds between checks")
    p.add_argument("--n-seeds", type=int, default=18)
    p.add_argument("--max-hours-per-arm", type=float, default=6.5,
                   help="per policy; four policies at 6.5h is ~26h total")
    p.add_argument("--lambda-multiplier", type=float, default=8.0,
                   help="second risk aversion, as a multiple of the calibrated one. The "
                        "default spans drift-dominated to risk-dominated: measured drift "
                        "rewards delay roughly as strongly as the calibrated penalty "
                        "punishes it, so a clearly higher lambda is needed to see "
                        "Almgren-Chriss front-loading emerge")
    p.add_argument("--skip-stop", action="store_true",
                   help="do not terminate the orchestrator (use if it already exited)")
    args = p.parse_args()

    if not wait_for_training(args.poll):
        return 1
    if not args.skip_stop:
        stop_session()

    if not os.path.exists(TRAIN_LOG):
        log(f"{TRAIN_LOG} missing -- cannot refit. Stopping.")
        return 1

    lam = calibrated_lambda()
    lam_high = lam * args.lambda_multiplier

    log("=" * 70)
    log("PHASE 2a: offline diagnostics and lambda sweep (no GPU, seconds)")
    log("=" * 70)
    run([sys.executable, "-u", "-m", "rl_execution.analyze_logs", TRAIN_LOG, "--last-run"])
    run([sys.executable, "-u", "-m", "rl_execution.impact", TRAIN_LOG])
    run([sys.executable, "-u", "-m", "rl_execution.refit_qtable", TRAIN_LOG,
         "--alpha-mode", "visit-count", "--stability",
         "--penalty-sweep", f"0,{lam:.2f},{lam_high:.2f}"])

    tables = {}
    for name, value in (("lam", lam), ("lamhigh", lam_high)):
        out = f"checkpoints/qtable_{name}.npz"
        if run([sys.executable, "-u", "-m", "rl_execution.refit_qtable", TRAIN_LOG,
                "--alpha-mode", "visit-count", "--inventory-penalty", f"{value:.4f}",
                "--out", out]) and os.path.exists(out):
            tables[name] = out
        else:
            log(f"refit at lambda={value:.2f} produced nothing; skipping that arm")

    if not tables:
        log("no refit Q-table available; falling back to the live-trained table")
        if os.path.exists(LIVE_QTABLE):
            tables["live"] = LIVE_QTABLE

    log("=" * 70)
    log("PHASE 2b: frontier evaluation -- TWAP, Almgren-Chriss, RL at each lambda")
    log("=" * 70)
    # Each Q-table is a separate invocation into the SAME log file and the same
    # eval-seed, so every policy is measured on an identical held-out set and the
    # comparison stays paired. TWAP and the AC schedule are re-run per invocation,
    # which is deliberate: it costs episodes but gives a within-invocation control.
    ok_any = False
    for i, (name, path) in enumerate(tables.items()):
        log(f"--- frontier arm {i + 1}/{len(tables)}: {name} ({path}) ---")
        ok_any |= run([
            sys.executable, "-u", "-m", "rl_execution.evaluate",
            "--n-seeds", args.n_seeds, "--skip-ddpm",
            "--max-hours-per-arm", args.max_hours_per_arm,
            "--ac-kappa", 2.0, "--qtable", path,
            "--ckpt-path", args.ckpt_path,
            "--out", f"logs/eval_frontier_{name}.jsonl",
        ])

    log("=" * 70)
    log("PHASE 2 COMPLETE" if ok_any else "PHASE 2 FINISHED WITH ERRORS -- check above")
    log("frontier logs: logs/eval_frontier_*.jsonl")
    log("analyse with:  python -m rl_execution.analyze_logs <log> --by policy_name")
    log("=" * 70)
    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
