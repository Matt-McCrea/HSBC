"""
Open-loop sampler evaluation: sample the TRADES model on REAL conditioning windows
(no ABIDES feedback loop) and compare generated order distributions against the real
next events for the same windows.

This is the attribution instrument for the frozen-mid-price investigation: if a sampler
config shows depth collapse / type imbalance HERE, the problem is the model x sampler
combination itself; if open-loop looks healthy but the closed-loop simulation still
freezes, the problem is the simulator's conditioning feedback.

Usage (on the GPU machine, from repo root):
    python evaluation/diagnostics/open_loop_eval.py --type DDPM --nsteps 100
    python evaluation/diagnostics/open_loop_eval.py --type DDIM --nsteps 10 --eta 0.0
    python evaluation/diagnostics/open_loop_eval.py --type DDIM --nsteps 10 --eta 1.0 \
        --n-windows 2048 --batch 256 --out results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

import constants as cst
import configuration
from utils.utils_data import load_compute_normalization_terms


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", required=True,
                    help="Sampler: DDPM, DDIM, DPM_SOLVER, DPM_SOLVER_PP, HYBRID_PP_DDIM, HYBRID_PP_DDPM")
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--tail-steps", type=int, default=2)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--id", type=float, default=None, help="checkpoint val loss (default: best)")
    ap.add_argument("--stock", type=str, default="INTC")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    ap.add_argument("--n-windows", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=str, default=None, help="JSON output path")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify checkpoint/dataset paths and arg wiring, skip model sampling")
    return ap.parse_args()


def find_checkpoint(symbol, wanted_val_loss):
    """Same discovery logic as world_agent_sim.py: best (lowest) val loss, or exact --id."""
    dir_path = Path(cst.DIR_SAVED_MODEL) / cst.Models.TRADES.value
    best_val_loss, checkpoint_reference = np.inf, None
    for file in dir_path.iterdir():
        if symbol not in file.name:
            continue
        try:
            val_loss = float(file.name.split("=")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if wanted_val_loss is not None:
            if val_loss == wanted_val_loss:
                checkpoint_reference = file
        elif val_loss < best_val_loss:
            best_val_loss, checkpoint_reference = val_loss, file
    if checkpoint_reference is None:
        raise FileNotFoundError(f"No matching checkpoint for {symbol} in {dir_path}")
    return checkpoint_reference


def load_model(checkpoint_reference, sampler_type, nsteps, eta, tail_steps, guidance_scale):
    from models.diffusers.diffusion_engine import DiffusionEngine
    torch.serialization.add_safe_globals(
        [configuration.Configuration, cst.Models, cst.LearningHyperParameter, cst.Stocks, cst.Engine])
    checkpoint = torch.load(checkpoint_reference, map_location=cst.DEVICE, weights_only=False)
    config = checkpoint["hyper_parameters"]["config"]
    config.IS_WANDB = False
    config.CHOSEN_MODEL = cst.Models.TRADES
    config.SAMPLING_TYPE = sampler_type
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_ETA] = eta
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_NSTEPS] = nsteps
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_TAIL_STEPS] = tail_steps
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.GUIDANCE_SCALE] = guidance_scale
    model = DiffusionEngine.load_from_checkpoint(checkpoint_reference, config=config, map_location=cst.DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, config


def denorm_stats(symbol):
    terms = load_compute_normalization_terms(symbol, cst.DATA_DIR, cst.Models.TRADES, n_lob_levels=10)
    e = terms["event"]
    return {"mean_size": e[0], "std_size": e[1], "mean_time": e[4], "std_time": e[5],
            "mean_depth": e[6], "std_depth": e[7]}


def decode_generated(x_t, type_anchors, size_type_emb, stats):
    """Decode a batch of raw model outputs (B, 1, size_order_emb) with the SAME logic as
    WorldAgent._postprocess_generated_TRADES. Returns dict of arrays."""
    g = x_t[:, 0, :]  # (B, F): [time, type_emb x3, size, price, direction, depth]
    emb = g[:, 1:size_type_emb + 1]
    # type: L1 nearest anchor (the original decode)
    d_l1 = torch.cdist(emb, type_anchors, p=1)
    types_l1 = torch.argmin(d_l1, dim=1).cpu().numpy()          # 0=limit,1=cancel,2=market
    d_l2 = torch.cdist(emb, type_anchors, p=2)
    types_l2 = torch.argmin(d_l2, dim=1).cpu().numpy()
    # prior-corrected decode: argmin(0.5*||x-anchor||^2 - log prior). Mirrors WorldAgent's
    # 'prior' type_decode so open-loop numbers are directly comparable.
    log_prior = torch.log(torch.tensor([0.49, 0.48, 0.03], device=emb.device))
    score = 0.5 * (d_l2 ** 2) - log_prior.unsqueeze(0)
    types_prior = torch.argmin(score, dim=1).cpu().numpy()
    size = np.round(g[:, size_type_emb + 1].cpu().numpy() * stats["std_size"] + stats["mean_size"])
    depth = np.round(g[:, -1].cpu().numpy() * stats["std_depth"] + stats["mean_depth"])
    time = g[:, 0].cpu().numpy() * stats["std_time"] + stats["mean_time"]
    direction = np.where(g[:, size_type_emb + 3].cpu().numpy() < 0, -1, 1)
    return {"type_l1": types_l1, "type_l2": types_l2, "type_prior": types_prior, "size": size,
            "depth": depth, "time": time, "direction": direction}


def decode_real(x_0, stats):
    """Decode real next events (B, 1, 6): [time, class, size, price, direction, depth] (z-scored)."""
    r = x_0[:, 0, :].cpu().numpy()
    return {
        "type": r[:, 1].astype(int),          # already class index 0/1/2
        "size": np.round(r[:, 2] * stats["std_size"] + stats["mean_size"]),
        "depth": np.round(r[:, 5] * stats["std_depth"] + stats["mean_depth"]),
        "time": r[:, 0] * stats["std_time"] + stats["mean_time"],
        "direction": np.where(r[:, 4] < 0, -1, 1),
    }


def summarize(name, types, sizes, depths, times, directions):
    n = len(types)
    type_hist = {"limit": float((types == 0).mean()),
                 "cancel": float((types == 1).mean()),
                 "market": float((types == 2).mean())}
    depth_hist = {"neg": float((depths < 0).mean()),
                  "0": float((depths == 0).mean()),
                  "1": float((depths == 1).mean()),
                  "2": float((depths == 2).mean()),
                  "3-5": float(((depths >= 3) & (depths <= 5)).mean()),
                  "6+": float((depths >= 6).mean())}
    return {
        "name": name, "n": n,
        "type_hist": type_hist,
        "depth_hist": depth_hist,
        "size_mean": float(np.mean(sizes)), "size_std": float(np.std(sizes)),
        "time_mean": float(np.mean(times)), "time_std": float(np.std(times)),
        "buy_share": float((directions == 1).mean()),
    }


def print_summary(s):
    th, dh = s["type_hist"], s["depth_hist"]
    print(f"  {s['name']:<22} n={s['n']}")
    print(f"    type   limit={th['limit']:.1%} cancel={th['cancel']:.1%} market={th['market']:.1%}")
    print(f"    depth  neg={dh['neg']:.1%} 0={dh['0']:.1%} 1={dh['1']:.1%} 2={dh['2']:.1%} "
          f"3-5={dh['3-5']:.1%} 6+={dh['6+']:.1%}")
    print(f"    size   mean={s['size_mean']:.0f} std={s['size_std']:.0f}   "
          f"time mean={s['time_mean']:.4f}s std={s['time_std']:.4f}   buy={s['buy_share']:.1%}")


def main():
    args = parse_args()

    checkpoint_reference = find_checkpoint(args.stock, args.id)
    print(f"checkpoint: {checkpoint_reference}")

    data_path = os.path.join(cst.DATA_DIR, args.stock, f"{args.split}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found — run preprocessing first")
    print(f"data      : {data_path}")

    if args.dry_run:
        print("dry run OK (checkpoint + dataset found, args parsed)")
        return

    model, config = load_model(checkpoint_reference, args.type, args.nsteps,
                               args.eta, args.tail_steps, args.guidance_scale)
    seq_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
    gen_seq_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE]
    size_type_emb = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB]

    from preprocessing.LOBDataset import LOBDataset
    dataset = LOBDataset(paths=[data_path], seq_size=seq_size,
                         gen_seq_size=gen_seq_size, chosen_model=cst.Models.TRADES)
    n_avail = len(dataset)
    n = min(args.n_windows, n_avail)
    # evenly-spaced window indices — decorrelates the heavily-overlapping stride-1 windows
    indices = np.linspace(0, n_avail - 1, n).astype(int)
    print(f"windows   : {n} of {n_avail} available (evenly spaced)")

    stats = denorm_stats(args.stock)
    type_anchors = model.type_embedder.weight.data  # (3, size_type_emb)

    gen_parts, real_parts = [], []
    with torch.no_grad():
        for start in range(0, n, args.batch):
            batch_idx = indices[start:start + args.batch]
            conds, x0s, lobs = zip(*(dataset[i] for i in batch_idx))
            cond_orders = torch.stack(conds).to(cst.DEVICE)
            x_0_real = torch.stack(x0s).to(cst.DEVICE)
            cond_lob = torch.stack(lobs).to(cst.DEVICE)
            x = torch.zeros(len(batch_idx), gen_seq_size, cst.LEN_ORDER,
                            device=cst.DEVICE, dtype=torch.float32)
            x_t = model.sample(cond_orders=cond_orders, x=x, cond_lob=cond_lob)
            gen_parts.append(decode_generated(x_t, type_anchors, size_type_emb, stats))
            real_parts.append(decode_real(x_0_real, stats))
            done = min(start + args.batch, n)
            print(f"  sampled {done}/{n}", flush=True)

    gen = {k: np.concatenate([p[k] for p in gen_parts]) for k in gen_parts[0]}
    real = {k: np.concatenate([p[k] for p in real_parts]) for k in real_parts[0]}

    tag = f"{args.type}_{args.nsteps}_eta{args.eta}"
    results = {
        "config": {"type": args.type, "nsteps": args.nsteps, "eta": args.eta,
                   "tail_steps": args.tail_steps, "guidance_scale": args.guidance_scale,
                   "checkpoint": str(checkpoint_reference), "n_windows": n, "split": args.split},
        "real": summarize("REAL next-events", real["type"], real["size"],
                          real["depth"], real["time"], real["direction"]),
        "generated_l1_decode": summarize(f"{tag} (L1 decode)", gen["type_l1"], gen["size"],
                                         gen["depth"], gen["time"], gen["direction"]),
        "generated_l2_decode": summarize(f"{tag} (L2 decode)", gen["type_l2"], gen["size"],
                                         gen["depth"], gen["time"], gen["direction"]),
        "generated_prior_decode": summarize(f"{tag} (prior decode)", gen["type_prior"], gen["size"],
                                            gen["depth"], gen["time"], gen["direction"]),
    }

    print(f"\n=== OPEN-LOOP RESULTS: {tag} ===")
    for key in ("real", "generated_l1_decode", "generated_l2_decode", "generated_prior_decode"):
        print_summary(results[key])

    out = args.out or f"open_loop_{tag}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
