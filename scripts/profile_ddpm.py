"""
profile_ddpm.py — where does DDPM's 100-step inference actually spend its time?

Uses torch.profiler (the standard PyTorch profiling tool — gives per-operation CPU/CUDA time,
finer-grained than manual breakpoint timing and without needing to step through 100 iterations by
hand). gaussian_diffusion.py's ddpm_sample/ddpm_single_step are labelled with
torch.profiler.record_function() markers for three phases, so the summary reads as clean phase
names instead of a wall of raw op names:
  ddpm_augment          — the conditioning-augmentation step, once per diffusion step
  ddpm_NN_forward       — the transformer forward pass (the actual generative model call)
  ddpm_loss_computation — VLB/MSE loss (only active during training; ddpm_sample passes
                          compute_loss=False during generation, so this should show ~0 here —
                          confirms the dead-work fix removed at the same time as this script)

Runs on real conditioning windows (same harness as evaluation/diagnostics/open_loop_eval.py), with
a few untimed warmup calls first (first-call CUDA kernel compilation otherwise skews the average).

Usage:
    python scripts/profile_ddpm.py --id 0.656 --n-calls 5
    python scripts/profile_ddpm.py --id 0.656 --n-calls 5 --out ddpm_trace.json
    # view the trace: copy ddpm_trace.json to a machine with a browser, open
    # https://ui.perfetto.dev and load it, or chrome://tracing in Chrome.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.profiler import profile, ProfilerActivity

import constants as cst
from evaluation.diagnostics.open_loop_eval import find_checkpoint, load_model


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=float, default=None, help="checkpoint val loss (default: best)")
    ap.add_argument("--stock", default="INTC")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--batch", type=int, default=32, help="batch size for the profiled sample() calls")
    ap.add_argument("--n-calls", type=int, default=5, help="number of sample() calls to profile+average")
    ap.add_argument("--warmup", type=int, default=2, help="untimed calls first (excludes CUDA compile overhead)")
    ap.add_argument("--out", default=None, help="chrome trace JSON path (optional)")
    return ap.parse_args()


def main():
    args = parse_args()
    checkpoint_reference = find_checkpoint(args.stock, args.id)
    print(f"checkpoint: {checkpoint_reference}")

    model, config = load_model(checkpoint_reference, "DDPM", 100, 0.0, 2, 1.0)
    seq_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE]
    gen_seq_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE]

    from preprocessing.LOBDataset import LOBDataset
    data_path = os.path.join(cst.DATA_DIR, args.stock, f"{args.split}.npy")
    dataset = LOBDataset(paths=[data_path], seq_size=seq_size, gen_seq_size=gen_seq_size,
                         chosen_model=cst.Models.TRADES)
    conds, x0s, lobs = zip(*(dataset[i] for i in range(args.batch)))
    cond_orders = torch.stack(conds).to(cst.DEVICE)
    cond_lob = torch.stack(lobs).to(cst.DEVICE)
    x = torch.zeros(args.batch, gen_seq_size, cst.LEN_ORDER, device=cst.DEVICE, dtype=torch.float32)

    print(f"batch={args.batch}  warmup={args.warmup} (untimed)  n_calls={args.n_calls} (profiled)")

    with torch.no_grad():
        for _ in range(args.warmup):
            model.sample(cond_orders=cond_orders, x=x, cond_lob=cond_lob)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
            for _ in range(args.n_calls):
                model.sample(cond_orders=cond_orders, x=x, cond_lob=cond_lob)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"

    print("\n" + "=" * 78)
    print(f"TOP 15 BY {key.upper()} (raw op-level — every low-level tensor op)")
    print("=" * 78)
    print(prof.key_averages().table(sort_by=key, row_limit=15))

    print("\n" + "=" * 78)
    print("PHASE SUMMARY (the three labelled regions, aggregated over all steps/calls)")
    print("=" * 78)
    avgs = prof.key_averages()
    phases = ["ddpm_augment", "ddpm_NN_forward", "ddpm_single_step", "ddpm_loss_computation"]
    total_cuda = sum(getattr(a, "self_cuda_time_total", 0) for a in avgs) or 1
    total_cpu = sum(a.self_cpu_time_total for a in avgs) or 1
    for name in phases:
        for a in avgs:
            if a.key == name:
                cuda_us = getattr(a, "self_cuda_time_total", 0)
                print(f"  {name:<24} calls={a.count:<8} "
                      f"self_cpu={a.self_cpu_time_total/1000:.2f}ms ({100*a.self_cpu_time_total/total_cpu:.1f}%)  "
                      f"self_cuda={cuda_us/1000:.2f}ms ({100*cuda_us/total_cuda:.1f}%)")
                break
        else:
            print(f"  {name:<24} not found in trace "
                  f"(expected for ddpm_loss_computation during generation — compute_loss=False)")

    if args.out:
        prof.export_chrome_trace(args.out)
        print(f"\nsaved chrome trace: {args.out}")
        print("view: copy to a machine with a browser, open https://ui.perfetto.dev (or "
              "chrome://tracing) and load the file — gives a visual timeline, closest thing to "
              "stepping through with breakpoints without actually doing it 100 times by hand.")


if __name__ == "__main__":
    main()
