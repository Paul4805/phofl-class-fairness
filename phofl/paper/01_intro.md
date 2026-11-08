# 1. Introduction

Federated learning (FL) fairness is usually reported as a single number: the
standard deviation of per-client test accuracy, often called FM. FM answers a
useful question — are clients treated equally? — but it is a one-dimensional
projection of a two-dimensional object. The other axis is class: a model can
equalize client accuracies while leaving entire semantic categories badly
served. This paper argues that client-FM alone is an incomplete reporting
metric for model-heterogeneous FL (MH-FL), and demonstrates the omission on a
controlled set of existing runs.

The headline is a **44.3-point best-to-worst class recall spread** in the
PHP-FL/linear run: frog 69.4% versus dog 25.1%. The run with the single best
client-FM (PHP-FL/entropy, FM=0.0763) is not better on this axis; its spread is
45.6 points. Thus the best client-level result is also a severely class-
imbalanced result. The 44.3-point case is the conservative headline.

The class axis is not merely a restatement of the participation pattern.
Within a method, the class-level standard deviation changes by only 0.0015 for
PHP-FL and 0.0084 for Standalone when moving from the linear to the
entropy-correlated pattern. The class imbalance is structural for these models,
not a consequence of the particular participation sampler.

We make four reporting contributions, not a new training method:

1. A client-by-class fairness report (the 2-D matrix `R[i,c]`) with
   support-aware cells.
2. A class-axis summary: `class_std` across per-class mean recalls and
   worst-class recall.
3. A proposed reporting metric, **class-balanced FM** (`class_FM`), to be
   reported alongside client-FM rather than optimized as an objective.
4. Evidence that client-FM and class-std can rank the same runs in opposite
   orders. This is an observation on five runs; n=5 is insufficient to test
   whether the reversal is systematic.

The scope of the field-wide claim was pre-registered. The FedAPEN/linear run
completed with a **52.9-point** class gap, above the 30-point threshold, so the
claim is promoted to field-wide across the two MH-FL methods tested (PHP-FL and
FedAPEN). This is not a universal claim about all FL methods; it is a
field-wide claim over the methods in this study. Between 20 and 30 points the
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
