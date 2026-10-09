# 01 — A degenerate-metric checklist

> Before you report any "improvement", run these four checks. Each one is a
> question you can answer in a few minutes, and each one can kill a result that
> looks convincing.

This document is deliberately **generic**. It does not describe any particular
project, model, or experiment. It describes four ways a metric can announce a
capability that is not there, and the action that exposes each one.

---

## Check 1 — The zero-ability question

**Definition.** Before you trust a metric as evidence for a capability, ask:
*"If the capability I am claiming were exactly zero, what would this metric
read?"* If the answer is "still high", the metric cannot be evidence for that
capability.

**When you step on it.** A "top-1 retrieval accuracy" of 98% sounds like a
strong retriever. If you never measured what a retriever with *no* ability
scores on the same task, 98% tells you nothing — the number could be 98% for a
random model, because the task itself is 98% easy (class imbalance, many
near-duplicates, or a metric that is satisfied trivially).

**How to check it.** Compute the metric for a *capability-free* system on the
exact same data, the exact same split, and the exact same metric:

- a majority-class / most-frequent predictor,
- a random scorer,
- the *untrained* or *frozen* version of your own model pushed through the
  identical pipeline.

Report the **gap**, not the raw number. The gap is the result. If the gap is
inside the noise of the baseline, you have measured the baseline, not the
capability.

---

## Check 2 — The identity / tautology exclusion

**Definition.** Check whether the metric has a *mathematically forced* value
along some path — a path where the two things being compared are not merely
correlated but are the **same computation**, a copy of the same tensor, or two
views of the same value. Such a comparison is guaranteed to be maximal, and the
metric cannot distinguish ability from arithmetic.

**When you step on it.** You compute a "query" representation and a "library
key" representation and measure their cosine similarity. If, for the matched
pair, both come from the **same text at the same position in the same forward
pass**, then — under a causal model — they are literally the same tensor.
Their cosine is exactly `1.0`, *even for a randomly initialized model*. A
top-1 accuracy built on this is `1.000` by construction. The metric is an
identity: it is re-reporting `x == x`.

This generalizes well beyond cosine. Anywhere the two sides of a comparison can
be traced back to one shared computation, the metric is degenerate:

- a value compared against a copy of itself,
- a target that was read from the same place as the prediction,
- a "held-out" set that overlaps the fit,
- a score defined so that its maximum is reached by construction.

**How to check it.** For the matched/relevant pair, ask two questions and
answer them by *tracing the code*, not by intuition:

1. **Same source?** Are both vectors derived from the same part of the input?
2. **Same computation?** Do they pass through the same operation on the same
   tensor at the same position, or are they computed from different inputs /
   different positions?

If both answers are "yes", the perfect value is arithmetic, not evidence.
Then **perturb one side**: change only the position, or only the text, and
re-measure. If the metric collapses from perfect to chance, the perfection
came from the identity.

See [`02_toy_identity_metric.md`](02_toy_identity_metric.md) for a runnable
version of exactly this check.

---

## Check 3 — Positional / semantic parity

**Definition.** The two things you compare must be *comparable*: the same kind
of object, read at a position and in a role that correspond. If the two sides
are not positionally or semantically on equal footing, similarity between them
has no interpretation.

**When you step on it.** Comparing the representation of a *whole document*
against the representation of a *different document's first token*, or
comparing a representation taken at position 0 against one taken at the last
position, produces numbers that are real but meaningless — they mix the effect
you care about with an effect you did not control (position, token role,
sequence length, padding).

A related failure: comparing a pooled summary of one sequence against a
single-token state of another. The vectors live in the same space, so a
similarity *exists*; it just does not measure what the label claims.

**How to check it.** Write down, for each side of the comparison, the tuple
`(which text, which position, how it is pooled, what role it plays)`. The two
tuples must match in every
field that is not the thing you are intentionally varying. If they differ in a
field you did not mean to vary, either control for it or state it explicitly in
the report.

---

## Check 4 — End-to-end consistency (title vs. path)

**Definition.** Confirm that the path you *actually executed* is the path the
title claims you tested. A result whose label says `X` but whose code computes
`Y` is not a weak result about `X`; it is a result about `Y` with a wrong
label.

**When you step on it.** The heading says "we evaluate retrieval", but the
actual call reads the ground-truth item directly out of the candidate set, or
the "held-out" evaluation silently reuses a cached embedding computed during
fitting, or the metric named in the table is computed by a function that does
something subtly different from the metric's usual definition.

These are usually not malicious. They are the normal entropy of a fast-moving
codebase: a name that outlived the code it named, a flag defaulting the wrong
way, a copy-paste that left the old target in place.

**How to check it.**

1. Read the metric's *implementation*, not its name. Confirm numerator and
   denominator are what the label says.
2. Add one **positive control and one negative control**: a configuration that
   *must* score high and one that *must* score at chance. If the negative
   control scores high, the pipeline is leaking.
3. Print the shapes and the actual tensors flowing into the metric for one
   batch, and eyeball that they are what the description claims.
4. Re-derive the headline number once by hand from the raw per-item outputs.

---

## How to use this checklist in practice

Run all four before writing any "improvement" into a summary. The order matters
because each check is cheap and they are roughly independent:

1. **Zero-ability** tells you whether the number can be evidence at all.
2. **Identity** tells you whether the number is forced by construction.
3. **Parity** tells you whether the comparison is interpretable.
4. **End-to-end** tells you whether you measured the thing on the label.

If any check fails, the honest move is not to delete the experiment. It is to
**record the failure and retract the claim** — see
[`03_ruled_out_and_retracted.md`](03_ruled_out_and_retracted.md). A caught
error is a deliverable; a silent one is a liability.
