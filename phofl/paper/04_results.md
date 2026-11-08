# 4. Results: the 44-point class spread

All results are from four completed half-scale CIFAR-10 runs: PHP-FL and
Standalone, each under linear and entropy-correlated participation. The runs
use 10 clients, 60 rounds, 1 seed, four CNNs, 1 local epoch, and 2,500 train /
500 test samples per client. Absolute accuracy is not comparable to the
paper's full-scale setting; the linear-vs-entropy contrast and the class-axis
geometry are the objects of interest.

## 4.1 Four-run summary

| method | pattern | client-FM | class-std | worst class | worst recall | best class | best recall | class gap |
|---|---|---:|---:|---|---:|---|---:|---:|
| PHP-FL | linear | 0.0785 | 0.1334 | dog | 0.251 | frog | 0.694 | **0.443** |
| PHP-FL | entropy | 0.0763 | 0.1349 | dog | 0.240 | truck | 0.696 | **0.456** |
| Standalone | linear | 0.0962 | 0.1281 | cat | 0.229 | frog | 0.615 | 0.386 |
| Standalone | entropy | 0.0985 | 0.1196 | cat | 0.218 | frog | 0.607 | 0.389 |

The conservative headline is the **44.3-point** spread in PHP-FL/linear:
frog 69.4% versus dog 25.1%. The single best client-FM run (PHP-FL/entropy,
FM=0.0763) is slightly worse on the class axis at 45.6 points. Reporting only
the best client-FM would therefore overstate the model's class fairness, not
understate it.

## 4.2 Per-class recall

| method | pattern | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PHP-FL | linear | 0.609 | 0.571 | 0.610 | 0.434 | 0.382 | 0.251 | 0.694 | 0.588 | 0.623 | 0.662 |
| PHP-FL | entropy | 0.626 | 0.613 | 0.599 | 0.457 | 0.397 | 0.240 | 0.673 | 0.511 | 0.616 | 0.696 |
| Standalone | linear | 0.405 | 0.586 | 0.564 | 0.229 | 0.335 | 0.285 | 0.615 | 0.381 | 0.488 | 0.541 |
| Standalone | entropy | 0.451 | 0.579 | 0.531 | 0.218 | 0.369 | 0.306 | 0.607 | 0.398 | 0.476 | 0.554 |

The class ordering is stable in its broad shape: dog and cat are the weakest
classes, frog/truck are among the strongest. The entropy pattern moves several
classes by a few points but does not remove the spread.

## 4.3 Support filtering

Each run has 100 client-by-class cells. **34 of 100 cells per run** (136 cells
total) have fewer than 10 test samples and are excluded from the worst-cell
calculation. The excluded cells are listed in `low_support_cells.csv`; they are
concentrated in airplane, truck, ship, cat, and dog, and include all 36
zero-sample cells.

The worst support-filtered cell is still 0.0 (client 0, cat, 11 samples) in
every run. It is sparse but above the threshold, and it is not the headline.
Computing class means only from supported client-cells increases the
PHP-FL/linear gap to **51.3 points** (worst class deer, 0.247) and the entropy
gap to 45.0 points. The global 44.3-point headline is therefore conservative.

## 4.4 Pattern invariance

Class-std is roughly invariant to the participation pattern within a method:

- PHP-FL: 0.1334 (linear) → 0.1349 (entropy), change **0.0015**.
- Standalone: 0.1281 (linear) → 0.1196 (entropy), change **0.0084**.

The class gap changes by about 1.3 points for PHP-FL and 0.3 points for
Standalone. The class imbalance is structural in these models; it is not
materially created or removed by the linear-vs-entropy sampler.

## 4.5 Ranking observation

Lower is better for both columns.

| run | client-FM | FM rank | class-std | class-std rank | delta |
|---|---:|---:|---:|---:|---:|
| PHP-FL / entropy | 0.0763 | 1 | 0.1349 | 4 | +3 |
| PHP-FL / linear | 0.0785 | 2 | 0.1334 | 3 | +1 |
| Standalone / linear | 0.0962 | 3 | 0.1281 | 2 | −1 |
| Standalone / entropy | 0.0985 | 4 | 0.1196 | 1 | −3 |

The two metrics ranked the four runs in opposite orders. This is an
observation, not a correlation claim: n=4 is insufficient to test whether the
reversal is systematic. The observation is still material because it shows
that the two axes can disagree on the same set of systems.

## 4.6 What this does and does not show

- It shows that a 44-point class spread coexists with a good client-FM.
- It shows that the class spread is stable across the two participation
  patterns.
- It shows that client-FM and class-std can order the same runs differently.
- It does **not** establish a field-wide claim by itself; Section 4b adds
  FedAPEN, which meets the pre-registered >30-point scope threshold.
- It does **not** revive the participation hypothesis, which failed its
  pre-registered escalation.

## 4b. FedAPEN

See `04_fedapen.md` for the completed FedAPEN/linear run and the
conditional claim-scope update.
