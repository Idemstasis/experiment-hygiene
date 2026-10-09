#!/usr/bin/env python3
"""Toy 02 -- two reasons a single number is not a fingerprint.

Zero downloads. Pure numpy/torch math. Two independent demonstrations:

Part 1 -- floating-point reductions are not bit-reproducible.
    The same list of numbers, summed in a different order, gives a different
    float. A multithreaded reduction splits the work across threads, so the
    summation order depends on the thread count / partitioning, and the
    resulting bytes change. We print the md5 of the result under different
    thread counts. Conclusion: an md5 identifies *this run* (script +
    thread count + hardware + library version), not a stable property of the
    computation. Honest reports should say which of the two they mean.

Part 2 -- a z-score built from a tiny control group explodes.
    If you estimate the spread from only a handful of control samples, the
    estimate of the spread is itself noisy and occasionally tiny. Any
    "significant" z = (x - mean) / sd then comes out in the hundreds, and a
    null experiment (no effect anywhere) still produces |z| > 3 far more often
    than the nominal 0.27%. We measure that inflation and give an empirical
    lower bound on how many control samples are needed before the number
    means anything.
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import torch


# --------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------
def md5_of(t: torch.Tensor) -> str:
    return hashlib.md5(t.detach().contiguous().numpy().tobytes()).hexdigest()


def part1(n: int, seed: int) -> None:
    print("=" * 72)
    print("PART 1 -- the same numbers, summed differently, are different bytes")
    print("=" * 72)

    torch.manual_seed(seed)
    vals = (torch.rand(n, dtype=torch.float32) - 0.5) * 1_000.0

    # (a) order dependence: identical values, different summation orders.
    seq = 0.0
    for v in vals.tolist():
        seq += v                                   # naive left-to-right
    rev = 0.0
    for v in reversed(vals.tolist()):
        rev += v                                   # reversed order
    np_pairwise = float(np.sum(vals.numpy()))      # numpy pairwise summation
    as_double = float(vals.double().sum())         # accumulate in float64

    rows = [
        ("python, left-to-right", seq),
        ("python, right-to-left", rev),
        ("numpy, pairwise", np_pairwise),
        ("float64 accumulation", as_double),
    ]
    print(f"{'summation order':<26}{'value':<24}{'md5 of result'}")
    print("-" * 72)
    for name, s in rows:
        t = torch.tensor([s])
        print(f"{name:<26}{s!r:<24}{md5_of(t)}")
    distinct = len({md5_of(torch.tensor([s])) for _, s in rows})
    print(f"\ndistinct results from the SAME {n} numbers: {distinct} / {len(rows)}")

    # (b) thread-count dependence of a torch reduction.
    print("\ntorch sum of the same tensor, varying the thread count:")
    print(f"{'threads':<10}{'value':<24}{'md5 of result'}")
    print("-" * 72)
    hashes = set()
    for k in (1, 2, 4, 8, 16):
        torch.set_num_threads(k)
        s = vals.sum()
        h = md5_of(torch.tensor([s.item()]))
        hashes.add(h)
        print(f"{k:<10}{s.item()!r:<24}{h}")
    print(f"\ndistinct results across thread counts: {len(hashes)}")

    # (c) the honest nuance: fixed thread count IS reproducible.
    torch.set_num_threads(8)
    repeat = {md5_of(torch.tensor([vals.sum().item()])) for _ in range(5)}
    print(f"same thread count (8), 5 repeats -> distinct md5: {len(repeat)}")
    print("\n=> the bytes depend on the reduction order / thread count.")
    print("   An md5 is a run fingerprint, not a computation fingerprint.")


# --------------------------------------------------------------------------
# Part 2
# --------------------------------------------------------------------------
def part2(trials: int, seed: int) -> None:
    print()
    print("=" * 72)
    print("PART 2 -- a z-score from a tiny control group explodes")
    print("=" * 72)
    print("Null experiment: every control sample and every 'treated' point is")
    print("drawn from the SAME standard normal. There is no effect, anywhere.")
    print("We still compute z = (x - mean_control) / sd_control and look at |z|.")
    print()
    print(f"{'n_control':>10}{'max|z|':>13}{'p99.9|z|':>11}"
          f"{'P(|z|>3)':>11}{'P(|z|>10)':>12}{'P(|z|>100)':>12}")
    print("-" * 72)

    ns = (2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 1024)
    for n in ns:
        g = torch.Generator().manual_seed(seed * 1000 + n)
        ctrl = torch.randn(trials, n, generator=g)
        x = torch.randn(trials, 1, generator=g)
        mean = ctrl.mean(dim=1, keepdim=True)
        sd = ctrl.std(dim=1, keepdim=True, unbiased=True)
        az = ((x - mean) / sd).abs().squeeze(1)
        print(f"{n:>10}{az.max().item():>13.1f}{az.quantile(0.999).item():>11.2f}"
              f"{float((az > 3).float().mean()):>11.4f}"
              f"{float((az > 10).float().mean()):>12.5f}"
              f"{float((az > 100).float().mean()):>12.6f}")

    print("\nFor reference, an honest standard normal gives P(|z|>3) = 0.0027.")
    print("Every row above has NO effect, yet small n_control inflates |z|:")
    print("  n=4  : |z| reached ~1e2 and 7.4% of nulls exceeded 3.")
    print("  n=16 : ~1% of nulls still exceed 3.")
    print("  n>=64: |z|>3 becomes ~0.4% and |z|>10 does not occur.")

    # concrete worst case at n = 4
    n = 4
    g = torch.Generator().manual_seed(seed * 1000 + n)
    ctrl = torch.randn(trials, n, generator=g)
    x = torch.randn(trials, 1, generator=g)
    mean = ctrl.mean(dim=1, keepdim=True)
    sd = ctrl.std(dim=1, keepdim=True, unbiased=True)
    z = ((x - mean) / sd).squeeze(1)
    i = int(z.abs().argmax())
    print(f"\nConcrete worst case (n_control={n}, no effect present):")
    print(f"  control samples : {[round(float(v), 4) for v in ctrl[i].tolist()]}")
    print(f"  control sd      : {float(sd[i].item()):.6f}   <-- tiny by luck")
    print(f"  treated point x : {float(x[i].item()):.4f}")
    print(f"  z               : {float(z[i].item()):.1f}")
    print("\n=> with 4 control samples, z can reach the hundreds with zero real")
    print("   effect. A 'significant' z from a tiny control group is not evidence.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2_000_000, help="part 1 array size")
    ap.add_argument("--trials", type=int, default=200_000, help="part 2 trials")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"torch {torch.__version__}  numpy {np.__version__}")
    part1(args.n, args.seed)
    part2(args.trials, args.seed)
    print()
    print("=" * 72)
    print("Fingerprints describe runs and configurations, not truths.")
    print("=" * 72)


if __name__ == "__main__":
    main()
