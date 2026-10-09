# 04 — Not bit-reproducible: why an md5 is a run fingerprint

Two independent reasons a single number (or a single hash) is weaker evidence
than it looks. Both are demonstrated by a zero-download toy:

```bash
python toy/02_z_score_explosion.py
```

---

## Part 1 — Floating-point reductions are order-dependent

Floating-point addition is **not associative**. `(a + b) + c` and `a + (b + c)`
can differ in the last bits. A multi-threaded reduction splits a sum across
threads and combines the partial results, so the summation order — and
therefore the exact bytes of the answer — depends on how the work was
partitioned.

The complete, unedited stdout of the script is reproduced in the two blocks
below: every printed line is present, split at the blank line between the
script's two parts (that separating line itself belongs to neither block).
Nothing inside either block is trimmed.

Part 1 (CPU-only, `torch 2.13.0+cpu`, `numpy 2.5.1`):

```
torch 2.13.0+cpu  numpy 2.5.1
========================================================================
PART 1 -- the same numbers, summed differently, are different bytes
========================================================================
summation order           value                   md5 of result
------------------------------------------------------------------------
python, left-to-right     74995.81923913956       19cba733f2c8dddf3fa20b18e238e170
python, right-to-left     74995.81923913956       19cba733f2c8dddf3fa20b18e238e170
numpy, pairwise           74995.65625             dfccb744fd77968e7576780f41343d12
float64 accumulation      74995.81923913956       19cba733f2c8dddf3fa20b18e238e170

distinct results from the SAME 2000000 numbers: 2 / 4

torch sum of the same tensor, varying the thread count:
threads   value                   md5 of result
------------------------------------------------------------------------
1         74995.765625            d8daae7bcb137d2dc383f03cdacdd3bd
2         74995.875               893f687bc04c60f2f62e8836610c8580
4         74995.9296875           1fd3343de905ba2c3f7858a90a8579bf
8         74995.78125             0654d2738a5b111e88714e41cf6813fd
16        74995.8359375           57d99408a48a907b3cbb2d65dd54f276

distinct results across thread counts: 5
same thread count (8), 5 repeats -> distinct md5: 1

=> the bytes depend on the reduction order / thread count.
   An md5 is a run fingerprint, not a computation fingerprint.
```

What this shows, exactly:

- The **same 2,000,000 numbers** give two different answers (`74995.65625` vs
  `74995.81923913956`) depending on the summation order — an absolute gap of
  about `0.16` at a magnitude of `7.5e4`, i.e. a handful of ulps in the
  accumulation but a **completely different md5**.
- A torch reduction gives **five different results / five different md5s** for
  thread counts 1, 2, 4, 8, 16.
- **And the honest nuance:** with the thread count held fixed, five repeats
  give **one** md5. The reduction is deterministic *given its configuration*.

### What this means for "reproducibility"

An md5 of a floating-point result is a fingerprint of the whole configuration
that produced it: the script, the thread count, the hardware, the library
version, and the partitioning. It is **not** a fingerprint of the mathematical
result, because the result has no single byte representation once a reduction
is involved.

So when you write "the output is bit-identical" or "the md5 matches", be
explicit about which of these two you mean:

1. **"This run is reproducible on this exact configuration."** True and useful —
   the fixed-thread repeat shows it.
2. **"This computation has a stable fingerprint."** False in general, and the
   thread sweep shows it.

Honest reporting writes (1) with the configuration pinned, and does not claim
(2). It also says plainly that a *different* number of threads, or a busy
machine that schedules differently, can change the bytes and hence any hash
derived from them.

> **Practical rule.** Never present an md5 as evidence of correctness. At most
> it is evidence that *this configuration* produced *this* artifact. If you
> need a stable fingerprint, hash the inputs and the version metadata, not the
> floating-point output.

---

## Part 2 — A z-score from a tiny control group explodes

The second trap is statistical, not numerical. Suppose you estimate the spread
of a control group from only a handful of samples, then form

```
z = (x − mean_control) / sd_control
```

The estimate `sd_control` is itself noisy, and for small `n` it is occasionally
*very* small. Dividing by an occasional near-zero inflates `z` without bound —
even when there is no effect anywhere.

The toy draws every control sample and every "treated" point from the **same
standard normal**. Nothing is real. Then it looks at `|z|` over 200,000 trials:

```
========================================================================
PART 2 -- a z-score from a tiny control group explodes
========================================================================
Null experiment: every control sample and every 'treated' point is
drawn from the SAME standard normal. There is no effect, anywhere.
We still compute z = (x - mean_control) / sd_control and look at |z|.

 n_control       max|z|   p99.9|z|   P(|z|>3)   P(|z|>10)  P(|z|>100)
------------------------------------------------------------------------
         2     753814.1     842.79     0.2454     0.07670    0.007845
         3       1335.4      36.48     0.1217     0.01290    0.000130
         4        118.3      14.16     0.0745     0.00287    0.000005
         5         74.7       9.62     0.0508     0.00090    0.000000
         8         13.9       5.63     0.0250     0.00002    0.000000
        16          6.3       4.18     0.0105     0.00000    0.000000
        32          5.8       3.72     0.0057     0.00000    0.000000
        64          5.1       3.50     0.0043     0.00000    0.000000
       128          4.6       3.41     0.0034     0.00000    0.000000
       256          5.6       3.36     0.0032     0.00000    0.000000
      1024          4.7       3.32     0.0028     0.00000    0.000000

For reference, an honest standard normal gives P(|z|>3) = 0.0027.
Every row above has NO effect, yet small n_control inflates |z|:
  n=4  : |z| reached ~1e2 and 7.4% of nulls exceeded 3.
  n=16 : ~1% of nulls still exceed 3.
  n>=64: |z|>3 becomes ~0.4% and |z|>10 does not occur.

Concrete worst case (n_control=4, no effect present):
  control samples : [-0.6036, -0.6, -0.5855, -0.5928]
  control sd      : 0.008006   <-- tiny by luck
  treated point x : 0.3516
  z               : 118.3

=> with 4 control samples, z can reach the hundreds with zero real
   effect. A 'significant' z from a tiny control group is not evidence.

========================================================================
Fingerprints describe runs and configurations, not truths.
========================================================================
```

Read the first rows as a warning:

- With **2** control samples, `|z|` exceeded **750,000**, and 24.5% of null
  trials exceeded 3.
- With **4** control samples, `|z|` reached **118**, and 7.4% of null trials
  exceeded 3.
- With **8**, 2.5% still exceeded 3 — about nine times the nominal rate.
- Even at **64**, the empirical `P(|z|>3)` is 0.43%, still above the 0.27% a
  standard normal would give, and it only approaches it slowly.

The concrete worst case shows the mechanism directly: the four control values
happened to land within `0.018` of each other, so `sd_control ≈ 0.008`; the
treated point, an ordinary draw, then scores `z ≈ 118`.

### An empirical lower bound on "how many control samples"

From the measured table (200,000 null draws per row):

| If you want, under a *null*, to see… | …you need roughly |
|---|---|
| `\|z\| > 100` not occur | `n_control ≥ 5` (it still occurred at 4) |
| `\|z\| > 10` to be rare (< 1 in 10⁵) | `n_control ≥ 16` |
| the empirical `P(\|z\|>3)` to be within ~2× of 0.27% | `n_control ≥ 64` |

There is deliberately no row for `|z| > 5`. The run above never prints a
`P(|z|>5)` column, and `max|z|` still exceeds 5 at `n_control = 256` (5.6), so
this run does not bound it — a few hundred control samples are not enough
either.

Treat these as **floors, not targets** — the exact numbers depend on the
distribution, but the shape is universal: below a few dozen control samples the
denominator is unstable, and any threshold you apply is really a threshold on
the luck of the control draw.

> **Practical rule.** If your control group has fewer than a few dozen samples,
> report the control samples themselves (all of them) and the raw effect size —
> *not* a standardized score. A "significant" `z` computed against a handful of
> control points is a statement about the control draw, not about the effect.

## One sentence

> Fingerprints describe runs and configurations. Small denominators describe
> luck. Neither is a truth about the world.
