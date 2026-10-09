# experiment-hygiene

**A short, runnable checklist for catching metrics that lie — and a discipline
for retracting honestly once they do.**

This repository is about one thing: *when does a number stop being evidence?*
It contains a four-item check you can run before reporting any "improvement",
two zero-download toys that make the failure modes concrete in a few seconds
each, and a procedure for withdrawing a claim without erasing it.

It does **not** propose a new method, and it is not a benchmark. It is the
hygiene that has to happen *before* a result is believed.

---

## The problem in one example

A metric reported **top-1 accuracy = 1.000**. The model behind it was randomly
initialized and had learned nothing. The number was real, reproducible, and
completely empty: the "query" and the matched "key" were **the same tensor
produced by the same computation**, so their similarity was `1.0` by
arithmetic. Move one of them to a different position and the metric drops to
chance. Nothing about the model changed.

That is the shape of the trap this repository is built around. It shows up in
many disguises, and it is cheap to catch — if you ask the right four questions.

---

## What's here

```
README.md                              this file
docs/
  01_degenerate_metric_check.md        the four checks (definition + when + how)
  02_toy_identity_metric.md            worked case: a "retrieval accuracy" that is an identity
  03_ruled_out_and_retracted.md        how to confirm, withdraw, and keep a retraction
  04_not_bit_reproducible.md           md5 is a run fingerprint; tiny-control z explodes
toy/
  README.md                            how to run the toys
  01_identity_metric.py                zero download: the identity metric, demonstrated
  02_z_score_explosion.py              zero download: reduction order + z-score explosion
requirements.txt
LICENSE                                MIT
```

## Quickstart

```bash
pip install -r requirements.txt
python toy/01_identity_metric.py
python toy/02_z_score_explosion.py
```

Both toys are **zero-download** and CPU-only. Toy 01 builds a tiny *randomly
initialized* causal transformer with `transformers.AutoConfig` +
`AutoModelForCausalLM.from_config` — no weights are fetched. Toy 02 is pure
`numpy`/`torch` math. Neither runs a backward pass through a language-model
head.

---

## The four checks

Full versions with examples and actions are in
[`docs/01_degenerate_metric_check.md`](docs/01_degenerate_metric_check.md).

| # | Check | Question | If it fails |
|---|---|---|---|
| 1 | **Zero-ability** | If the claimed capability were exactly zero, what would this metric read? | If "still high", the metric cannot be evidence for that capability. |
| 2 | **Identity exclusion** | Do the two sides trace back to the *same computation* (same tensor, same position, same pass)? | The perfect value is arithmetic, not ability. Perturb one side and re-measure. |
| 3 | **Positional / semantic parity** | Are the two sides the same kind of object, in corresponding roles and positions? | The similarity exists but does not mean what the label says. |
| 4 | **End-to-end consistency** | Is the path that actually ran the path the claim names? | You measured `Y` under the label `X`; fix the label or the pipeline. |

## The retraction discipline

Full version in [`docs/03_ruled_out_and_retracted.md`](docs/03_ruled_out_and_retracted.md).

1. State the suspect as a **falsifiable claim** with a number.
2. **Two reviewers, independently** — no cross-talk before both are done.
3. Take the **union** of their findings (either flag ⇒ suspect), never the
   intersection.
4. Write a four-field record for each retracted item
   (`claim / status / reason / replacement`).
5. **Keep the retraction at the top of the evidence list**, dated. Do not
   delete it.

A retraction is not a confession; it is the output of the machine working. An
evidence bundle that leads with what it caught is stronger than one that lists
only clean numbers.

## The two reproducibility facts

Full version, with verbatim output, in
[`docs/04_not_bit_reproducible.md`](docs/04_not_bit_reproducible.md).

- **A floating-point reduction is order-dependent.** The same 2,000,000 numbers
  summed in different orders (or across different thread counts) produce
  different bytes and therefore different md5s. An md5 is a *run*
  fingerprint, not a *computation* fingerprint.
- **A z-score from a tiny control group explodes.** With 4 control samples and
  no effect whatsoever, `z` reached **118** in a 200,000-trial null; 7.4% of
  null trials exceeded `|z| > 3` against a nominal 0.27%. Below a few dozen
  control samples, report the raw effect, not a standardized score.

---

## Prior work

The four checks below are **not new ideas**, and we do not claim they are. We
independently arrived at the same conclusions this literature already
establishes; the contribution of this repository is deliberately small and
different: a **runnable minimal reproduction**, a **checklist you can paste
into a report**, and a **retraction procedure** to go with them.

**Shortcut learning.** Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge &
Wichmann, *Shortcut Learning in Deep Neural Networks*, arXiv:2004.07780 (2020),
published in *Nature Machine Intelligence*, DOI 10.1038/s42256-020-00257-z.
Shortcuts are decision rules that score well on a benchmark without capturing
the intended capability. This is the general form of "the metric went up but
the ability did not". <https://arxiv.org/abs/2004.07780>

**Measured nulls for self-improvement claims.** Xu, Yan, Chen & Kechadi,
*Phantom Gains: Auditing Self-Improvement Against a Measured Null*,
arXiv:2608.20290 (2026). Tracking per-problem gains and losses differences two
noisy estimates; without a null *measured on the same pipeline*, the machinery
manufactures capability changes on a frozen model. Their headline example: an
expansion statistic that reports a **0.280** rate for a model that never
trained. This is exactly the "measure what no-ability scores" discipline of
Check 1. <https://arxiv.org/abs/2608.20290>

**The floor under cosine similarity.** Ethayarajh, *How Contextual are
Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and
GPT-2 Embeddings*, arXiv:1909.00512 (EMNLP 2019). Learned representations are
**anisotropic**: they occupy a narrow cone rather than filling the space, so
even two different words can carry a high average cosine similarity. So the
cosine of two *unrelated* items is not near zero, and an
absolute threshold does not transfer. This bears directly on the off-diagonal
cosine and the random-pair floor visible in
[`docs/02`](docs/02_toy_identity_metric.md). <https://arxiv.org/abs/1909.00512>

**Cosine similarity can be arbitrary.** Steck, Ekanadham & Kallus, *Is
Cosine-Similarity of Embeddings Really About Similarity?*, arXiv:2403.05440
(WWW 2024 Companion), DOI 10.1145/3589335.3651526. Cosine similarity of learned
embeddings can "yield arbitrary and therefore meaningless 'similarities'",
controlled by regularization choices rather than by the data. Another reason a
cosine-based metric needs a measured baseline before it is read.
<https://arxiv.org/abs/2403.05440>

**Independently-established, and not separately cited here.** The
majority-class / dummy-predictor baseline is textbook practice and needs no
single reference; it is the everyday form of Check 1.

> Note on attribution: every reference above was checked against its arXiv
> listing before inclusion — the abstract for the claim it is cited for, and the
> paper body for any passage quoted verbatim. Anything we could not verify was
> removed rather than stated from memory.

---

## What this is / is not

- **Is:** a short checklist, two runnable toys, and a retraction procedure. All
  generic; no project-specific mechanism, location, or protocol appears
  anywhere in this repository.
- **Is not:** a novel method, a benchmark, or a claim about any particular
  model. It does not report any new capability result, because it has none to
  report.
- **Is not** a guarantee. The toys demonstrate specific, narrow failure modes
  on specific configurations. They are teaching artifacts, not proofs, and
  passing these checks does not make a result correct — it only means it
  survived four cheap filters.
- The numbers in the docs are the **verbatim output** of the scripts in `toy/`
  on the configuration recorded there. Nothing is transcribed from memory.

---

## License

MIT — see [`LICENSE`](LICENSE).
