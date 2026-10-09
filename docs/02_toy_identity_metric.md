# 02 — Toy case: a "retrieval accuracy" that is really an identity

This is the runnable companion to **Check 2** (identity exclusion) in
[`01_degenerate_metric_check.md`](01_degenerate_metric_check.md). It takes an
abstract trap and makes it concrete in a few seconds, with **zero downloads**.

Run it:

```bash
python toy/01_identity_metric.py
```

---

## The setup

We build a tiny **randomly initialized** causal transformer — `gpt2` config via
`transformers.AutoConfig` + `AutoModelForCausalLM.from_config`, a few layers,
small width, **no weights downloaded**. It has learned nothing.

We build a "library" of 64 random token sequences and ask a "retrieval"
question: *for each query, is its nearest library key the intended one?*
Formally, top-1 accuracy = fraction of queries whose maximum-cosine key is the
key they were paired with. Chance is `1/64 ≈ 0.0156`.

## The trap

The query and the matched library key are the **same text**, read at the
**same position (the last token)**, on the **same forward pass**. So they are
not similar — they are the *same tensor* computed once. Under a causal model
nothing downstream can change that.

## What actually happened (verbatim output)

```
Toy 01 -- a 'retrieval accuracy' that is really an identity
------------------------------------------------------------------------
model          : randomly initialized (no weights downloaded)
               : gpt2 config, vocab=256, d=32, layers=2, heads=4
library size M : 64
sequence length: 16
chance level   : 1/M = 0.0156
device         : cpu   torch=2.13.0+cpu

========================================================================
A. DEGENERATE: query IS a library item, read at the SAME position
========================================================================
query == key tensor elementwise? : True
matched-pair cosine              : min=1.000000 max=1.000000  (exactly 1.0 expected)
off-diagonal cosine              : n=4032  min=-0.3829  mean=0.2836  max=0.9968  p50=0.2932
top-1 'accuracy'                 : 1.0000   <-- 1.000
interpretation: 1.000 here is the identity, not retrieval ability.

========================================================================
B. CONTROL: same texts & model, only the POOLED POSITION changes
========================================================================
query==key tensor elementwise?    : False
max cosine per query             : n=64  min=0.2146  mean=0.5052  max=0.7651  p50=0.5243
top-1 'accuracy'                 : 0.0156   (chance=0.0156)
interpretation: breaking the identity drops the metric to (near) chance.

========================================================================
C. CONTROL: query and keys are DIFFERENT texts
========================================================================
max cosine per query             : n=64  min=0.5019  mean=0.7209  max=0.9945  p50=0.6706
top-1 'accuracy'                 : 0.0312   (chance=0.0156)
interpretation: unrelated texts carry no identity; metric ~ chance.

========================================================================
Permutation test on variant C (what 'no relationship' scores)
========================================================================
observed top-1 accuracy : 0.0312
null distribution       : mean=0.0154 sd=0.0152  p95=0.0469 max=0.0938
p-value (>= observed)   : 0.2484

========================================================================
SUMMARY
========================================================================
A degenerate (same text, same position) : 1.0000
B control    (same text, other position): 0.0156
C control    (different texts)          : 0.0312
chance                                  : 0.0156

The metric read 1.000 only while the query and the matched key were
literally the same tensor from the same computation. Change one
position and it tells the truth about a random model: ~chance.
```

(Output captured from `torch 2.13.0+cpu`, CPU-only, Python 3.14.4. The toy
fixes all seeds, so re-runs on this configuration reproduce it exactly; see
[`04_not_bit_reproducible.md`](04_not_bit_reproducible.md) for why "exactly"
deserves an asterisk.)

## How to read the three variants

- **A (degenerate).** `query == key` is `True` elementwise, the matched cosine
  is exactly `1.000000` for every pair, and the "accuracy" is `1.0000`. The
  model is random; there is no retrieval ability. The `1.000` is the identity
  `x == x`, nothing more.
- **B (control, isolates the cause).** Same texts, same model, same forward
  pass — only the *pooled position* of the query changes (first token instead
  of last). Now `query == key` is `False`, the cosine is no longer 1, and the
  metric collapses to `0.0156`, exactly chance. This is the key experiment: the
  *only* thing that changed was the identity, and the metric followed it.
- **C (control, different texts).** Query and keys are unrelated texts. The
  metric reads `0.0312`, i.e. chance within noise, and the permutation test
  gives `p = 0.2484`: indistinguishable from "no relationship".

So the metric read `1.000` **only** in the variant where the two sides were the
same computation, and dropped to chance the moment that was broken. That is the
signature of a degenerate metric.

## Two side observations

**The off-diagonal cosine is not near zero (anisotropy).** In variant A the
*unmatched* pairs average cosine `0.2836`, and in variant C unrelated texts
reach a maximum of `0.9945`. The floor of "two unrelated things" is high and
depends on the model and the corpus. This is a known property of learned
embeddings (see Prior work in the root [`README.md`](../README.md)); the
practical consequence is that an absolute cosine threshold is not portable and
must be calibrated against a measured random-pair floor.

**The permutation test is the zero-ability question made numerical.** Shuffling
which query is paired with which key gives the null distribution of the metric.
The observed value sits inside it. Note that this null is *measured on the same
pipeline* — the point made at length in the "Phantom Gains" audit cited in the
README.

## The one-line lesson

> A comparison between a thing and a copy of itself is guaranteed to be
> maximal. Before you read a perfect score, check that the two sides are not
> the same computation.
