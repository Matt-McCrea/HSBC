"""
check_checkpoint_configs.py — which checkpoints came from which training run/config?

Filenames only encode val_ema + epoch, NOT which run produced them. But each .ckpt saves its full
training config internally (checkpoint["hyper_parameters"]["config"]), so we can check directly
whether CONDITIONAL_DROPOUT was on (the retrain that enables classifier-free guidance) rather than
guess from epoch numbers. This matters: mixing checkpoints from different runs under one "checkpoint
quality bracket" sweep would confound val-loss-quality with run-identity.

Usage:
    python scripts/check_checkpoint_configs.py [/path/to/checkpoint/dir]
    (defaults to cst.DIR_SAVED_MODEL/TRADES if no path given)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import constants as cst
import configuration


def main():
    ckpt_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(cst.DIR_SAVED_MODEL) / cst.Models.TRADES.value
    files = sorted(ckpt_dir.glob("*.ckpt"))
    if not files:
        print(f"no .ckpt files found in {ckpt_dir}")
        return

    torch.serialization.add_safe_globals(
        [configuration.Configuration, cst.Models, cst.LearningHyperParameter, cst.Stocks, cst.Engine])

    rows = []
    for f in files:
        try:
            val_loss = float(f.name.split("=")[1].split("_")[0])
            epoch = int(f.name.split("epoch=")[1].split("_")[0])
        except (IndexError, ValueError):
            val_loss, epoch = float("nan"), -1
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        config = ckpt["hyper_parameters"]["config"]
        hp = config.HYPER_PARAMETERS
        dropout = hp.get(cst.LearningHyperParameter.CONDITIONAL_DROPOUT, None)
        seq_size = hp.get(cst.LearningHyperParameter.SEQ_SIZE, None)
        lr = hp.get(cst.LearningHyperParameter.LEARNING_RATE, None)
        mtime = f.stat().st_mtime
        rows.append((epoch, val_loss, dropout, seq_size, lr, mtime, f.name))

    rows.sort(key=lambda r: r[0])  # by epoch
    print(f"{'epoch':>5} {'val_ema':>8} {'dropout':>9} {'seq_size':>9} {'lr':>10}  {'mtime':>19}   file")
    print("-" * 100)
    import datetime
    for epoch, val_loss, dropout, seq_size, lr, mtime, name in rows:
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{epoch:>5} {val_loss:>8.3f} {str(dropout):>9} {str(seq_size):>9} {str(lr):>10}  {ts:>19}   {name}")

    dropouts = {r[2] for r in rows}
    print()
    if len(dropouts) > 1:
        print(f"!! MIXED CONFIGS DETECTED — CONDITIONAL_DROPOUT values seen: {dropouts}")
        print("   These checkpoints are NOT all from the same training run. Any 'checkpoint quality")
        print("   bracket' sweep across them confounds val-loss with run-identity — treat comparisons")
        print("   across differing-dropout checkpoints with caution.")
    else:
        print(f"All checkpoints share CONDITIONAL_DROPOUT={dropouts.pop()} — consistent single-run config.")


if __name__ == "__main__":
    main()
