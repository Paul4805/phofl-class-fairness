# 6. Discussion

## 5.1 A reporting practice, not a method

The result is a critique of a reporting convention, not a proposal for a new
federated-learning algorithm. We do not introduce a new aggregation rule, a new
local objective, or a new participation sampler. The claim is narrower: an MH-FL
evaluation that reports only client-FM is missing a first-order property of the
model.

We recommend that MH-FL papers report a minimal two-axis table:

```
client_FM
class_FM          (std over classes of per-class mean recall)
worst_class_recall
```

plus the client-by-class matrix or a link to it. If space is tight, report
client-FM, class_FM, and worst-class recall. The three numbers are cheap to
compute from logs that are already collected for a per-client evaluation.

`class_FM` is a **reporting metric**, not an objective. Optimizing it directly
would create the same kind of blind spot in the other direction: a model could
equalize class means while sacrificing client-level performance or the worst
client. It should be reported alongside client-FM, not used to replace it.

## 5.2 Why the class axis is structural

The invariance of class-std to the participation pattern is the most useful
negative observation in this study. Moving from linear to entropy-correlated
participation changes class-std by 0.0015 for PHP-FL and 0.0084 for Standalone,
while the class gap remains tens of points. This suggests that the class
imbalance is carried by the model/architecture/data interaction, not by the
particular client-selection schedule.

That does not mean participation is irrelevant to class fairness in every
setting. It means that in these four runs, changing the pattern did not
materially change the class axis. A larger, multi-seed study is needed to test
whether that invariance is robust.

## 5.3 Ranking inversion is an observation

The five runs disagree on client-FM and class-std, and FedAPEN is 3rd by
client-FM but 5th by class gap. This is a striking observation, and it is
tempting to describe the two metrics as anti-correlated. We deliberately do
not. With five runs, a rank correlation is descriptive at best; it cannot
establish a systematic relationship. The defensible statement is: “the two
metrics ordered these five runs differently; n=5 is insufficient to test
whether this is systematic.”

The observation still matters for practice: a leaderboard sorted by client-FM
can put a class-imbalanced model first. The converse is also possible in a
larger run set. This is precisely why both axes should be reported.

## 5.4 Limitations

- **Scale.** The half-scale runs use 1 local epoch, 60 rounds, 10 clients, four
  CNNs, and 2,500/500 samples per client. This is roughly 8x less training than
  the paper's full setting. Absolute AM/FM are not comparable to the paper.
- **Seeds.** Three seeds for the eight-cell comparison (four methods × two
  patterns). The ladder remains single-seed. Seed sd is reported wherever the
  class axis is compared, and it is large enough that only FedAPEN's ordering
  is established.
- **Methods.** Eight cells cover PHP-FL, FedAPEN, FedFV, and Standalone.
  FedAPEN/linear has a 0.60 ± 0.09 class gap, above the pre-registered
  30-point threshold in every seed, so the claim is field-wide across the two
  MH-FL methods tested (PHP-FL and FedAPEN). It is not universal across all FL
  methods. FedFV is a reimplementation applied to the shared model, and FedHEAL
  is not implemented.
- **Support.** Per-client class cells are sparse. We use a pre-declared
  >=10-sample threshold and report the excluded cells. The global class-level
  headline uses the full 1,000-sample test set per class, so it is not driven by
  the sparse cells.
- **AM confound.** The single-seed gap/AM adjustment is superseded: with three
  seeds PHP-FL and Standalone are tied on the raw class gap, so there is no
  Standalone advantage to normalize away. The PHP-FL-vs-FedAPEN rank inversion
  survives AM normalization (FedAPEN is 3rd by client-FM and 5th by gap/AM).
- **Causality.** The analysis is descriptive. It does not identify a causal
  mechanism for the class spread, and it does not test the closed
  participation hypothesis.

## 5.5 What would change the conclusion

The class-fairness claim would be weakened if a larger, multi-seed, full-scale
study showed that the 44-point spread shrinks to a few points, or that
class-std tracks client-FM across methods. It would be strengthened if the
client-FM/class-std order reversal persists across seeds and a wider method
set. FedAPEN already meets the pre-registered >30-point scope threshold; that
is a scope promotion, not a confirmation of the rank-inversion hypothesis.
