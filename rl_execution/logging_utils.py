"""Shared JSON-lines logging schema for benchmark/train/eval scripts (deliverable 7).

One JSON object per line, one line per episode, so the Results chapter can be
written directly from saved logs without rerunning anything (per the spec).
Every field that's cheap to capture is captured -- better to have an unused
column than to need a rerun to add one later.
"""

import json
import os
import time
import uuid


EPISODE_LOG_FIELDS = (
    "run_name", "run_id", "timestamp", "seed_day", "t0", "side", "Q", "sampling_type",
    "depth_noise", "ddim_nsteps", "checkpoint", "policy_name", "wall_clock_total_s",
    "wall_clock_reconstruct_s", "wall_clock_simulate_s", "p_arrival", "shortfall",
    "shortfall_bps", "reward", "n_resting_orders", "n_steps", "fills", "cond_z",
    "flow_mix", "execution_rate", "unique_mid_count", "error",
)


class JsonlLogger:
    """Append-only JSON-lines writer. One instance per run (benchmark run,
    training run, or eval run); one line written per episode via log_episode().
    """

    def __init__(self, path):
        self.path = path
        # One id per logger instance, i.e. per run. Logs are append-only and filenames get
        # reused across runs (a smoke test then the real run writing to the same file), so
        # without this the rows of several runs are indistinguishable and per-episode
        # numbering silently shifts -- which is exactly how a "look at episode 56" request
        # landed on a different episode entirely.
        self.run_id = uuid.uuid4().hex[:8]
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def log_episode(self, **fields):
        unknown = set(fields) - set(EPISODE_LOG_FIELDS)
        if unknown:
            raise ValueError(f"unknown log field(s): {unknown} -- add to EPISODE_LOG_FIELDS if intentional")
        record = {k: fields.get(k) for k in EPISODE_LOG_FIELDS}
        record["run_id"] = record["run_id"] or self.run_id
        record["timestamp"] = record["timestamp"] or time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        return record


def shortfall_bps(info):
    """Implementation shortfall in basis points of the arrival mid -- the unit the
    execution literature reports, and unit-free (raw shortfall is in LOBSTER price
    ticks, i.e. $1e-4, which is meaningless without p_arrival alongside it).
    """
    shortfall = info.get("shortfall")
    p_arrival = info.get("p_arrival")
    if shortfall is None or not p_arrival:
        return None
    return float(shortfall) / float(p_arrival) * 10_000.0


def _json_default(o):
    if hasattr(o, "tolist"):  # numpy scalars/arrays
        return o.tolist()
    return str(o)


def read_episodes(path):
    """Read a JSON-lines log file back into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
