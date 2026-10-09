# 03 — Ruled out, and retracted: a discipline for honest withdrawal

A checklist that only *finds* problems is half a tool. The other half is what
you do once you have found one. This document is the procedure: how to confirm
a suspect result, how to withdraw it without erasing it, and why the withdrawal
itself is evidence you should keep.

The procedure is generic. It contains no description of any specific project's
model, mechanism, or protocol.

---

## The procedure

### Step 0 — Name the suspect as a claim, not a feeling

Write the suspicious item as a single falsifiable sentence with a subject, a
metric, and a number, e.g.:

> "Configuration X reaches top-1 accuracy 1.00 on task T."

A claim can be reviewed. "Something feels off about X" cannot.

### Step 1 — Two independent reviews, no cross-talk

Have **two reviewers examine the claim separately**, without seeing each
other's findings and without discussing it first. Independence is the entire
point: if reviewer 2 reads reviewer 1's notes, you have one review with two
signatures, and correlated blind spots survive.

Each reviewer answers, independently:

1. Is the metric's maximum reachable by construction (identity / tautology)?
2. What does a capability-free system score on the same metric, split, and
   pipeline?
3. Are the two sides of the comparison positionally and semantically on equal
   footing?
4. Is the path that ran the same as the path the claim names?

Record each reviewer's verdict per item: **holds / suspect / refuted**, with
the reason.

### Step 2 — Take the union, not the intersection

An item is **suspect if either reviewer flagged it.** Do not require both to
agree before acting. Intersecting the two reviews throws away exactly the
findings you most need — the ones only one careful reader caught. The cost of
keeping a false positive in the suspect list is a few minutes of checking; the
cost of dropping a true positive is a published error.

### Step 3 — Retract each item and write it down

For each confirmed suspect item, produce a **retraction record** and store it.
A record has four fields:

```text
claim      : the claim as originally stated
status     : retracted / narrowed / superseded
reason     : which check failed, and the concrete evidence (command + output)
replacement: what, if anything, may still be said instead
```

Retraction does not have to mean "everything was wrong". Often the honest
outcome is **narrowing**: "1.00 on the identity path" becomes "1.00 is
arithmetic; on the corrected path the value is at chance". Say exactly which
part survives.

### Step 4 — Keep the retraction at the TOP of the evidence list

**Do not delete the withdrawn result and do not bury it.** Put the retraction
record at the *head* of the list of results, above the surviving ones, with the
date. A reader who only skims the top of your evidence file must see that a
claim was withdrawn and why.

Deleting a retracted result is the single most damaging move available here. It
destroys the audit trail, it hides the fact that the pipeline can be fooled,
and it makes the surviving results *less* trustworthy, not more — because now
no reader can tell whether those were checked too.

---

## The general lesson

> A retraction is not a confession. It is the output of the machine working.

If your process reports only findings that survived, a reader learns nothing
about whether the process can catch anything. If it reports *what it caught and
how*, the reader learns that the pipeline has been tested against its own
failure modes — which is strictly more information.

Concretely: an evidence bundle that leads with "we found a degenerate metric,
here is the identity that caused it, here is the corrected number" is a
stronger artifact than one that only lists clean numbers, because it
demonstrates the filter exists.

---

## An abstract worked example

*(Illustrative and generalized; it does not reproduce any particular study.)*

**Claim.** "Pipeline A reaches top-1 accuracy 1.00 on retrieval task T, so A
has established retrieval."

**Check 4 (end-to-end) fires first.** The heading says T, but reading the code,
the "query" is read out of the same candidate set that provides the "keys", in
the same pass, at the same position.

**Check 2 (identity) fires second.** For the matched pair, both vectors are the
same tensor from the same computation. Cosine is `1.0` by arithmetic. The
metric is `1.000` for any model, trained or not.

**Check 1 (zero-ability) settles it.** Pushing a *frozen / untrained* model
through the identical pipeline also scores `1.000`. The number is a property of
the pipeline, not of A.

**Retraction record.**

```text
claim      : A reaches top-1 accuracy 1.00 on T => retrieval established
status     : retracted
reason     : the query and the matched key are the same tensor from the same
             forward pass (check 2); an untrained model also scores 1.000
             (check 1); the label named the wrong measured path (check 4)
replacement: on a corrected pipeline (query and keys from different texts and
             positions) the score is at chance; no retrieval claim is supported
```

**Where it goes.** Filed at the top of the evidence list, dated, above the
results that survived. The corrected number, and the fact that the error was
caught, are both part of the deliverable.

---

## A minimal checklist for the withdrawal itself

- [ ] The suspect was written as a falsifiable claim with a number.
- [ ] Two reviewers examined it **independently**, before comparing notes.
- [ ] The suspect list is the **union** of their findings, not the intersection.
- [ ] Every retracted item has a four-field record (claim / status / reason / replacement).
- [ ] The records are **at the top** of the evidence list, dated, not deleted.
- [ ] For each record, the original claim is still quoted verbatim so it can be found.
