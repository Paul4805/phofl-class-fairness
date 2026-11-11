# 1. Introduction

Federated learning (FL) fairness is usually reported as a single number: the
standard deviation of per-client test accuracy, often called FM. FM answers a
useful question — are clients treated equally? — but it is a one-dimensional
projection of a two-dimensional object. The other axis is class: a model can
equalize client accuracies while leaving entire semantic categories badly
served. This paper argues that client-FM alone is an incomplete reporting
metric for model-heterogeneous FL (MH-FL), and demonstrates the omission on a
controlled set of existing runs.

The headline is a **0.39 ± 0.08 best-to-worst class recall spread** in
PHP-FL/linear over three seeds, with seed 0 at 0.468 (frog 69.4% versus dog
25.1%). The run with the single best client-FM (PHP-FL/entropy, FM=0.0763) is
not better on this axis. Thus the best client-level result is also a severely
class-imbalanced result. The earlier single-seed 44.3-point figure sits inside
that spread and is reported as a seed-0 draw, not as the headline.

The class axis is not merely a restatement of the participation pattern, but
the pattern effect is not uniform either. Re-running with three seeds shows the
linear-to-entropy change moves the class gap by up to 15 points, in a direction
that depends on the aggregation rule: it hurts FedFV (3/3) and Standalone
(3/3), is mixed for PHP-FL, and helps FedAPEN (3/3). The single-seed claim that
class imbalance is "structural / participation-agnostic" is therefore narrowed
to: the class gap is large under every pattern tested, and its size is
method-specific.

With three seeds the method ordering also changes. PHP-FL and Standalone are
indistinguishable on the class axis (0.387 vs 0.389, per-seed differences change
sign), while FedAPEN is consistently worse in 6/6 comparisons (0.60 vs 0.39).

We make four reporting contributions, not a new training method:

1. A client-by-class fairness report (the 2-D matrix `R[i,c]`) with
   support-aware cells.
2. A class-axis summary: `class_std` across per-class mean recalls and
   worst-class recall.
3. A proposed reporting metric, **class-balanced FM** (`class_FM`), to be
   reported alongside client-FM rather than optimized as an objective.
4. Evidence that client-FM and class-std can rank the same runs in opposite
   orders. With three seeds the worst-gap run is also the best-client-FM run in
   two of three seeds; the reversal is a replicated observation on these
   methods, not a universal law.

The scope of the field-wide claim was pre-registered. The FedAPEN/linear run
completed with a class gap of 0.60 ± 0.09, well above the 30-point threshold,
so the claim is promoted to field-wide across the two MH-FL methods tested
(PHP-FL and FedAPEN). This is not a universal claim about all FL methods; it is
a field-wide claim over the methods in this study. Between 20 and 30 points the
evidence would have been called mixed, and below 20 points the claim would have
narrowed to PHP-FL. Section 4b records the measured values and the rule.

Finally, the participation-specific hypothesis that motivated the original
experiment is closed: its pre-registered escalation failed (FM −2.84% relative,
min-accuracy +4.09 points, rho difference 0.170). We do not revisit or rescue
that hypothesis here. The contribution is a metric critique and a reporting
practice, not a causal claim about participation probabilities.

## Scope and non-claims

- The analysis is descriptive and post hoc over completed runs.
- The half-scale setting uses 1 local epoch, 60 rounds, 10 clients, four CNNs,
  2,500 train / 500 test per client. Absolute AM/FM are not comparable to the
  paper's full-scale CIFAR-10 numbers.
- We do not claim a new FL algorithm. `class_FM` is a reporting metric.
- We do not claim that the observed rank inversion is systematic; five runs
  cannot support that inference.
