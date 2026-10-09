# toys

Two zero-download, CPU-only scripts. Each takes a few seconds and prints
its evidence directly to stdout.

## Setup

```bash
pip install -r ../requirements.txt
```

Use the CPU-only build of `torch`. No model weights are downloaded and no
network access is used.

## `01_identity_metric.py` — a "retrieval accuracy" that is an identity

Builds a tiny **randomly initialized** causal transformer
(`transformers.AutoConfig` + `AutoModelForCausalLM.from_config`) and measures a
"top-1 retrieval accuracy" three ways:

- **A — degenerate:** the query and the matched library key are the same text,
  read at the same position, in the same forward pass. They are literally the
  same tensor, cosine is exactly `1.0`, and the metric is `1.000` **even though
  the model is random**.
- **B — control:** identical texts and model; only the pooled *position* of the
  query changes. The identity breaks and the metric collapses to chance.
- **C — control:** query and keys are different texts; also chance.

A permutation test gives the null distribution of the metric. The takeaway is
in [`../docs/02_toy_identity_metric.md`](../docs/02_toy_identity_metric.md).

```bash
python 01_identity_metric.py
python 01_identity_metric.py --n-docs 128 --seq-len 24 --seed 1
```

Options: `--seed --n-docs --seq-len --vocab-size --dim --layers --heads
--n-perm`.

## `02_z_score_explosion.py` — fingerprints and denominators

Two independent demonstrations:

- **Part 1:** the same numbers, summed in a different order (or across a
  different thread count), produce different bytes and different md5s. An md5
  is a *run* fingerprint, not a *computation* fingerprint. A fixed thread count
  is shown to be reproducible, so the claim is stated carefully.
- **Part 2:** with a tiny control group, `z = (x − mean) / sd` explodes on a
  null experiment (no effect anywhere). The script sweeps `n_control` from 2 to
  1024 and reports how often `|z|` exceeds 3 / 10 / 100, plus a concrete
  worst-case example.

Details in [`../docs/04_not_bit_reproducible.md`](../docs/04_not_bit_reproducible.md).

```bash
python 02_z_score_explosion.py
python 02_z_score_explosion.py --trials 50000 --seed 3
```

Options: `--n --trials --seed`.

## A note on running these

- Toy 01 performs **forward passes only**. It never calls a backward pass
  through the language-model head.
- All random seeds are fixed; on the recorded configuration the output is
  byte-stable, and Part 1 of toy 02 explains why that qualifier matters.
