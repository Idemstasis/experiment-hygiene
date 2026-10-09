#!/usr/bin/env python3
"""Toy 01 -- a "retrieval accuracy" metric that is secretly an identity.

Zero downloads: this script builds a tiny, RANDOMLY INITIALIZED causal
transformer (`transformers.AutoConfig` + `AutoModelForCausalLM.from_config`)
and never touches the network.

It demonstrates one narrow, general trap:

    If the "query" vector and the matched "library key" vector come from the
    SAME text, at the SAME position, on the SAME forward pass, then -- under a
    causal model -- they are not merely similar, they are literally the same
    tensor on the same computation. Their cosine similarity is exactly 1.0, so
    a "top-1 retrieval accuracy" built from those vectors is 1.000 by
    construction, no matter what the model has or has not learned.

Three variants are printed:

    A. Degenerate  : query and matched key are the same text, same position.
                     -> identical tensor, cosine == 1.0, top-1 == 1.000.
    B. Control     : same texts and same model, only the POOLED POSITION of
                     the query differs. The identity is broken and the metric
                     collapses toward chance.
    C. Control     : query and keys are different texts. Also ~chance.

B and C exist to isolate the cause of the 1.000: it is the identity of the two
vectors, not any retrieval capability. The model here is random, so a
*correctly posed* metric is expected to report chance -- which is the honest
answer, and the whole point.

A permutation test (shuffling the query->target assignment) gives the null
distribution of the metric, i.e. what "no relationship" scores.

This script never runs a backward pass through the language-model head.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


# --------------------------------------------------------------------------
# tiny random model
# --------------------------------------------------------------------------
def build_random_model(seed: int, vocab_size: int, seq_len: int,
                       dim: int, layers: int, heads: int):
    """A tiny causal LM with random weights. No download, no training."""
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "gpt2",
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_embd=dim,
        n_layer=layers,
        n_head=heads,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


@torch.no_grad()
def encode_last(model, ids: torch.Tensor) -> torch.Tensor:
    """Hidden state of the LAST position of every row. (B, dim)."""
    out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states[-1][:, -1, :]


@torch.no_grad()
def encode_pos(model, ids: torch.Tensor, pos: int) -> torch.Tensor:
    """Hidden state of a chosen position of every row. (B, dim)."""
    out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states[-1][:, pos, :]


def top1_accuracy(query: torch.Tensor, keys: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Fraction of queries whose nearest key is the intended one (index i)."""
    sims = F.normalize(query, dim=-1) @ F.normalize(keys, dim=-1).T  # (Q, M)
    pred = sims.argmax(dim=-1)
    target = torch.arange(query.shape[0])
    acc = (pred == target).float().mean().item()
    return acc, sims


def perm_test(query: torch.Tensor, keys: torch.Tensor, n_perm: int,
              seed: int) -> tuple[float, list[float]]:
    """Null distribution of top-1 accuracy under a shuffled assignment."""
    sims = F.normalize(query, dim=-1) @ F.normalize(keys, dim=-1).T
    pred = sims.argmax(dim=-1)
    target = torch.arange(query.shape[0])
    observed = (pred == target).float().mean().item()

    g = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_perm):
        perm = torch.randperm(target.numel(), generator=g)
        null.append((pred == target[perm]).float().mean().item())

    ge = sum(1 for v in null if v >= observed - 1e-12)
    p = (1 + ge) / (n_perm + 1)
    return observed, null


def describe(x: torch.Tensor) -> str:
    x = x.flatten()
    return (f"n={x.numel()}  min={x.min():.4f}  mean={x.mean():.4f}  "
            f"max={x.max():.4f}  p50={x.median():.4f}")


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-docs", type=int, default=64, help="library size M")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = build_random_model(args.seed, args.vocab_size, args.seq_len,
                               args.dim, args.layers, args.heads)

    g = torch.Generator().manual_seed(args.seed + 1)
    docs = torch.randint(0, args.vocab_size, (args.n_docs, args.seq_len),
                         generator=g)                       # the "library"
    others = torch.randint(0, args.vocab_size, (args.n_docs, args.seq_len),
                           generator=g)                     # unrelated texts

    chance = 1.0 / args.n_docs

    print("Toy 01 -- a 'retrieval accuracy' that is really an identity")
    print("-" * 72)
    print(f"model          : randomly initialized (no weights downloaded)")
    print(f"               : gpt2 config, vocab={args.vocab_size}, "
          f"d={args.dim}, layers={args.layers}, heads={args.heads}")
    print(f"library size M : {args.n_docs}")
    print(f"sequence length: {args.seq_len}")
    print(f"chance level   : 1/M = {chance:.4f}")
    print(f"device         : {torch.device('cpu')}   torch={torch.__version__}")

    # -- Variant A ---------------------------------------------------------
    banner("A. DEGENERATE: query IS a library item, read at the SAME position")
    key_vecs = encode_last(model, docs)      # library keys: last position
    q_vecs_a = encode_last(model, docs)      # query: last position, same text
    acc_a, sims_a = top1_accuracy(q_vecs_a, key_vecs)

    matched = (q_vecs_a * key_vecs).sum(dim=-1) / (
        q_vecs_a.norm(dim=-1) * key_vecs.norm(dim=-1))
    identical = torch.equal(q_vecs_a, key_vecs)
    print(f"query == key tensor elementwise? : {identical}")
    print(f"matched-pair cosine              : min={matched.min():.6f} "
          f"max={matched.max():.6f}  (exactly 1.0 expected)")
    offdiag = sims_a[~torch.eye(args.n_docs, dtype=torch.bool)]
    print(f"off-diagonal cosine              : {describe(offdiag)}")
    print(f"top-1 'accuracy'                 : {acc_a:.4f}   <-- 1.000")
    print("interpretation: 1.000 here is the identity, not retrieval ability.")

    # -- Variant B ---------------------------------------------------------
    banner("B. CONTROL: same texts & model, only the POOLED POSITION changes")
    q_vecs_b = encode_pos(model, docs, 0)    # query pooled at FIRST position
    acc_b, sims_b = top1_accuracy(q_vecs_b, key_vecs)
    cap_b = sims_b.max(dim=-1).values
    print(f"query==key tensor elementwise?    : {torch.equal(q_vecs_b, key_vecs)}")
    print(f"max cosine per query             : {describe(cap_b)}")
    print(f"top-1 'accuracy'                 : {acc_b:.4f}   (chance={chance:.4f})")
    print("interpretation: breaking the identity drops the metric to (near) chance.")

    # -- Variant C ---------------------------------------------------------
    banner("C. CONTROL: query and keys are DIFFERENT texts")
    q_vecs_c = encode_last(model, others)
    acc_c, sims_c = top1_accuracy(q_vecs_c, key_vecs)
    cap_c = sims_c.max(dim=-1).values
    print(f"max cosine per query             : {describe(cap_c)}")
    print(f"top-1 'accuracy'                 : {acc_c:.4f}   (chance={chance:.4f})")
    print("interpretation: unrelated texts carry no identity; metric ~ chance.")

    # -- permutation test --------------------------------------------------
    banner("Permutation test on variant C (what 'no relationship' scores)")
    obs, null = perm_test(q_vecs_c, key_vecs, args.n_perm, args.seed + 7)
    nt = torch.tensor(null)
    ge = sum(1 for v in null if v >= obs - 1e-12)
    print(f"observed top-1 accuracy : {obs:.4f}")
    print(f"null distribution       : mean={nt.mean():.4f} "
          f"sd={nt.std():.4f}  p95={nt.quantile(0.95):.4f} max={nt.max():.4f}")
    print(f"p-value (>= observed)   : {(1 + ge) / (args.n_perm + 1):.4f}")

    # -- takeaway ----------------------------------------------------------
    banner("SUMMARY")
    print(f"A degenerate (same text, same position) : {acc_a:.4f}")
    print(f"B control    (same text, other position): {acc_b:.4f}")
    print(f"C control    (different texts)          : {acc_c:.4f}")
    print(f"chance                                  : {chance:.4f}")
    print()
    print("The metric read 1.000 only while the query and the matched key were")
    print("literally the same tensor from the same computation. Change one")
    print("position and it tells the truth about a random model: ~chance.")


if __name__ == "__main__":
    main()
