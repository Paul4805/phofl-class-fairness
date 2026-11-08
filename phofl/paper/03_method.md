# 3. Two-axis fairness report

This section defines the reporting object. It is an analysis protocol, not a
new training algorithm.

## 3.1 Client-by-class matrix

For a run with clients `i = 1..K` and classes `c = 1..C`, let

```
R[i,c] = recall on class c for client i's test set
n[i,c] = number of test samples of class c held by client i
```

The matrix `R` is the two-dimensional fairness object. Client-level accuracy is
the row mean; class-level recall is the column mean, with an explicit choice of
client weighting. The report stores `R`, `n`, and both summaries.

## 3.2 Client axis

The existing client-FM is

```
client_FM = std_i( mean_c R[i,c] )
```

where the standard deviation is over clients. This is the quantity reported by
PHP-FL and related MH-FL work. It is useful but marginalizes the class axis.

## 3.3 Class axis

We report two class-axis summaries:

```
class_std = std_c( mean_i R[i,c] )
class_gap = max_c mean_i R[i,c] - min_c mean_i R[i,c]
```

and **worst-class recall**, the minimum of the per-class mean recalls. The
headline uses the canonical global per-class recall vector (the full CIFAR-10
test set, 1,000 samples per class), not the support-filtered client mean.
`class_std` is the population standard deviation unless stated otherwise.

The class-axis summaries are deliberately not collapsed into a single “fair”
score. A method can have a low `class_std` and a bad worst class, or vice
versa. We report `class_std`, `class_gap`, and worst-class recall together.

## 3.4 Support filtering

Per-client class cells can be extremely sparse. A client may hold fewer than
ten test examples of a class, making a recall of 0 or 1 uninformative. We use a
pre-declared threshold of **10 test samples per (client, class) cell**:

- cells with `n[i,c] >= 10` are retained for the worst-cell calculation;
- cells below the threshold are listed in `low_support_cells.csv` and excluded
  from the headline and from the support-filtered sensitivity analysis.

In the four half-scale runs, 34 of 100 cells per run fall below the threshold
(136 cells total, including 36 zero-sample cells). The worst supported cell is
still 0.0 (client 0, cat, 11 samples), so support filtering does not explain
away the worst cell. The class-level headline is based on the full global test
set and is unaffected by the per-client support filter; the support-filtered
class-mean spread is reported as a sensitivity check.

## 3.5 Proposed reporting metric: class-balanced FM

We propose **class-balanced FM** as a reporting metric:

```
class_FM = std_c( mean_i R[i,c] )
```

Lower is better. It is the class-axis analogue of client-FM. It should be
reported **alongside** client-FM and worst-class recall, not instead of them.
It is not a training objective: we do not optimize `class_FM`, and we do not
claim it is a sufficient fairness metric by itself. The recommendation is a
two-number report plus the worst-class tail:

```
(client_FM, class_FM, worst_class_recall)
```

## 3.6 Ranking analysis

For a set of runs, rank the runs separately by client-FM and by `class_std`
(lower is better for both). A disagreement means the two axes order the same
systems differently. On our five runs the orders still disagree (FedAPEN is 3rd by client-FM and
5th by class gap). With n=5 we report this only as an observation; we do not
attach a rank-correlation p-value or claim that the reversal is systematic.

## 3.7 What is not proposed

- No new aggregation rule.
- No new local loss.
- No new participation sampler.
- No claim that class-FM should replace client-FM.
- No causal claim about participation probabilities; that hypothesis is
  closed.
