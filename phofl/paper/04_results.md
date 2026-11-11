# 4. Results: the 44-point class spread

All results are from four completed half-scale CIFAR-10 runs: PHP-FL and
Standalone, each under linear and entropy-correlated participation. The runs
use 10 clients, 60 rounds, 1 seed, four CNNs, 1 local epoch, and 2,500 train /
500 test samples per client. Absolute accuracy is not comparable to the
paper's full-scale setting; the linear-vs-entropy contrast and the class-axis
geometry are the objects of interest.

## 4.1 Seed-0 summary (historical)

| method | pattern | client-FM | class-std | worst class | worst recall | best class | best recall | class gap |
|---|---|---:|---:|---|---:|---|---:|---:|
| PHP-FL | linear | 0.0785 | 0.1334 | dog | 0.251 | frog | 0.694 | **0.443** |
| PHP-FL | entropy | 0.0763 | 0.1349 | dog | 0.240 | truck | 0.696 | **0.456** |
| Standalone | linear | 0.0962 | 0.1281 | cat | 0.229 | frog | 0.615 | 0.386 |
| Standalone | entropy | 0.0985 | 0.1196 | cat | 0.218 | frog | 0.607 | 0.389 |

These are the original single-seed numbers. They are retained for
reproducibility but are superseded by §4.1b: the 44.3-point PHP-FL gap is the
seed-0 draw from a distribution whose mean is 0.39.

## 4.1b Three-seed replication

Eight cells (four methods × two patterns) × three seeds, on one stack
(`torch 2.14.1`). Class gap is mean ± sample sd.

| method | pattern | AM | client-FM | class gap | worst-class recall |
|---|---|---|---|---|---|
| FedFV | linear | 0.6272 ± 0.0080 | 0.0874 ± 0.0035 | **0.3527 ± 0.0446** | **0.346 ± 0.039** |
| PHP-FL | linear | 0.6286 ± 0.0083 | 0.0860 ± 0.0095 | 0.3867 ± 0.0766 | 0.312 ± 0.053 |
| Standalone | linear | 0.5423 ± 0.0014 | 0.0996 ± 0.0095 | 0.3887 ± 0.0189 | 0.215 ± 0.009 |
| PHP-FL | entropy | 0.6302 ± 0.0041 | 0.0817 ± 0.0032 | 0.4083 ± 0.0170 | 0.278 ± 0.031 |
| Standalone | entropy | 0.5377 ± 0.0060 | 0.0954 ± 0.0100 | 0.4087 ± 0.0076 | 0.212 ± 0.014 |
| FedFV | entropy | 0.6262 ± 0.0087 | 0.0847 ± 0.0042 | 0.4330 ± 0.0304 | 0.268 ± 0.026 |
| FedAPEN | entropy | 0.6064 ± 0.0075 | 0.0873 ± 0.0109 | 0.5043 ± 0.0517 | 0.276 ± 0.052 |
| FedAPEN | linear | 0.6151 ± 0.0030 | 0.0779 ± 0.0137 | **0.6017 ± 0.0853** | 0.207 ± 0.019 |

Three consequences:

1. **Seed variance is first-order.** PHP-FL/linear spans 0.316–0.468 across
   seeds and FedAPEN/linear spans 0.547–0.700. A single-seed class gap is not a
   measurement, it is a draw.
2. **PHP-FL is not better than Standalone on the class axis.** Per-seed
   differences are +0.071, −0.051, −0.026 (linear) and +0.014, −0.008, −0.007
   (entropy): no consistent sign. The 9-point AM advantage does not translate
   into a class-axis advantage.
3. **FedAPEN is consistently worst** (6/6 per-seed comparisons against PHP-FL,
   +0.058 to +0.324). This is the only ordering that survives reseeding, and
   the field-wide claim rests on it.

FedFV/linear has the best mean gap and worst-class recall, but its 3.4-point
edge over PHP-FL is 3/3 in sign and smaller than PHP-FL's seed sd (two-sided
sign test p = 0.25). It is reported as a directional observation, not a fix.

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

## 4.4 Pattern effect is method-dependent

On seed 0, class-std looked almost invariant to the participation pattern
(PHP-FL +0.0015, Standalone −0.0084). The three-seed sweep shows the class gap
moving by up to 15 points, with a direction that depends on the method:

| method | entropy − linear (gap), per seed | reading |
|---|---|---|
| Standalone | +0.015, +0.033, +0.012 | worse, 3/3 |
| PHP-FL | −0.042, +0.076, +0.031 | mixed |
| FedFV | +0.069, +0.108, +0.064 | worse, 3/3 |
| FedAPEN | −0.037, −0.108, −0.147 | better, 3/3 |

The correct statement is: the class gap is large under both patterns, but the
entropy-correlated sampler is not neutral — it interacts with the aggregation
rule, helping FedAPEN and hurting FedFV/Standalone. The earlier "structural,
participation-agnostic" reading was a seed-0 artifact of small changes.

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

- It shows that a 0.35–0.60 class spread coexists with a good client-FM.
- It shows that the class spread is present under both participation patterns,
  with a method-dependent effect size rather than invariance.
- It shows that client-FM and class-std can order the same runs differently,
  and that the reversal repeats across seeds.
- It does **not** establish a field-wide claim by itself; Section 4b adds
  FedAPEN, which meets the pre-registered >30-point scope threshold.
- It does **not** revive the participation hypothesis, which failed its
  pre-registered escalation.

## 4b. FedAPEN

See `04_fedapen.md` for the completed FedAPEN/linear run and the
conditional claim-scope update.
