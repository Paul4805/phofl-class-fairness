# Class fairness: the 44-point spread that client-FM hides

**Headline:** The **44.3-point best-to-worst class recall spread** appears in
the PHP-FL/linear run, within the method family with the lowest client-FM. The
single best-FM run (PHP-FL/entropy, FM=0.0763) is slightly worse on the class
axis at 45.6 points, so 44.3 is the conservative case. The completed
FedAPEN/linear run has a **52.9-point** gap, so the finding is no longer
PHP-FL-specific.

All numbers are descriptive analyses of completed half-scale runs. No new
training was run for this document.

## 2-D fairness object

For a run with clients `i` and classes `c`, the natural object is
`R[i,c] = recall on class c for client i`. Client-FM is a one-dimensional
projection:

```
client_FM = std_i( mean_c R[i,c] )
```

The class axis is the other projection:

```
class_std = std_c( mean_i R[i,c] )
```

The client-by-class matrices are in `phofl/analysis/matrices/`; the tidy version
with test-sample counts is `two_axis_client_class.csv`; the client-level table
with class columns is `two_axis_fairness.csv`.

## Support filtering

Each run has 100 `(client, class)` cells. **34 of 100 cells per run (170 total
across the five runs)** have fewer than 10 test samples and are excluded from
the worst-cell calculation. They are listed in `low_support_cells.csv`; counts
are in `support_exclusions_summary.csv`. The excluded cells are concentrated in
airplane/truck/ship/cat/dog and include all 45 zero-sample cells.

After support filtering, the worst supported cell is still **0.0** — client 0,
cat, with 11 test samples, in every run. It is not a low-support artifact, but
it is still sparse and is **not** the headline. The headline is the class-level
44.3-point spread, whose per-class recalls are based on the full 1,000-sample
CIFAR-10 test set per class.

As a sensitivity check, computing class means only from supported client-cells
gives a **51.3-point** gap for PHP-FL/linear (worst class deer, 0.247) and a
**41.6-point** gap for FedAPEN/linear (worst class cat, 0.242). The
44.3-point global PHP-FL figure is the conservative version. See
`class_std_supported` and `supported_class_gap` in
`two_axis_run_summary.csv`.

## Five-run summary

| method | pattern | client_FM | class_std | worst class | worst recall | best class | best recall | class gap | support-filtered gap |
|---|---|---:|---:|---|---:|---|---:|---:|---:|
| PHPFL | linear | 0.0785 | 0.1334 | dog | 0.251 | frog | 0.694 | 0.443 | 0.513 |
| PHPFL | entropy_correlated | 0.0763 | 0.1349 | dog | 0.240 | truck | 0.696 | 0.456 | 0.450 |
| FedAPEN | linear | 0.0839 | 0.1698 | cat | 0.256 | frog | 0.785 | **0.529** | 0.416 |
| Standalone | linear | 0.0962 | 0.1281 | cat | 0.229 | frog | 0.615 | 0.386 | 0.350 |
| Standalone | entropy_correlated | 0.0985 | 0.1196 | cat | 0.218 | frog | 0.607 | 0.389 | 0.354 |

Per-class recall:

| method | pattern | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PHPFL | linear | 0.609 | 0.571 | 0.610 | 0.434 | 0.382 | 0.251 | 0.694 | 0.588 | 0.623 | 0.662 |
| PHPFL | entropy_correlated | 0.626 | 0.613 | 0.599 | 0.457 | 0.397 | 0.240 | 0.673 | 0.511 | 0.616 | 0.696 |
| FedAPEN | linear | 0.520 | 0.704 | 0.631 | 0.256 | 0.482 | 0.275 | 0.785 | 0.639 | 0.663 | 0.424 |
| Standalone | linear | 0.405 | 0.586 | 0.564 | 0.229 | 0.335 | 0.285 | 0.615 | 0.381 | 0.488 | 0.541 |
| Standalone | entropy_correlated | 0.451 | 0.579 | 0.531 | 0.218 | 0.369 | 0.306 | 0.607 | 0.398 | 0.476 | 0.554 |

## AM normalization

`AM_std` is the client standard deviation (numerically FM), not a seed-level
standard error. `gap_over_am` is `class_gap / AM`; `gap_over_ceiling` is
`class_gap / (1 - worst_class_recall)`.

| method | pattern | AM | AM_std | raw gap | gap/AM | gap/ceiling | support gap | support gap/AM | support gap/ceiling |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PHPFL | linear | 0.6320 | 0.0785 | 0.443 | 0.7010 | 0.5915 | 0.5125 | 0.8110 | 0.6803 |
| PHPFL | entropy | 0.6308 | 0.0763 | 0.456 | 0.7229 | 0.6000 | 0.4499 | 0.7132 | 0.6399 |
| FedAPEN | linear | 0.6209 | 0.0839 | 0.529 | 0.8519 | 0.7110 | 0.4159 | 0.6697 | 0.5484 |
| Standalone | linear | 0.5429 | 0.0962 | 0.386 | 0.7110 | 0.5006 | 0.3502 | 0.6450 | 0.4363 |
| Standalone | entropy | 0.5467 | 0.0985 | 0.389 | 0.7115 | 0.4974 | 0.3541 | 0.6477 | 0.4579 |

The AM-independent statement is the PHP-FL-vs-FedAPEN inversion: FedAPEN is
3rd by client-FM and 5th by gap/AM (0.852), so the inversion survives AM
normalization.

## Pattern invariance

Class-std is **roughly invariant to participation pattern within a method**
(the FedAPEN run is linear only, so it does not enter this comparison):

- PHPFL: 0.1334 (linear) vs 0.1349 (entropy), change **0.0015**.
- Standalone: 0.1281 (linear) vs 0.1196 (entropy), change **0.0084**.

The class gap changes by only ~1.3 points for PHPFL and ~0.3 points for
Standalone. This bias is structural: it is not moved materially by the
linear-vs-entropy participation pattern.

## FedAPEN and claim scope

FedAPEN/linear was the single queued follow-up run. Its result:

- client-FM 0.0839, class-std 0.1698, class gap **52.9 points**
  (cat 25.6% vs frog 78.5%), support-filtered gap 41.6 points.
- FedAPEN ranks **3rd by client-FM** but **5th (worst) by class gap**. Its
  ranking by client-FM and its ranking by class gap are inverted.

The pre-specified scope rule was: >30-point class gap → promote to field-wide;
<20 → narrow to PHP-FL; 20–30 → mixed. At **52.9 points**, FedAPEN falls in the
field-wide category. The defensible scope is:

> across the two MH-FL methods tested (PHP-FL and FedAPEN), client-FM can hide
> a >40-point class-recall spread.

This is not a universal claim about all FL methods. It is a field-wide claim
over the two MH-FL methods in this study, chosen by the pre-specified rule.
No “confirmation” language is used; FedAPEN **tests** the pre-registered scope
threshold and falls above it.

## Rankings still disagree

Lower is better for all ranks.

| run | client_FM | FM rank | class_std | class-std rank | class gap | class-gap rank |
|---|---:|---:|---:|---:|---:|---:|
| PHPFL / entropy | 0.0763 | 1 | 0.1349 | 4 | 0.456 | 4 |
| PHPFL / linear | 0.0785 | 2 | 0.1334 | 3 | 0.443 | 3 |
| FedAPEN / linear | 0.0839 | 3 | 0.1698 | 5 | 0.529 | 5 |
| Standalone / linear | 0.0962 | 4 | 0.1281 | 2 | 0.386 | 1 |
| Standalone / entropy | 0.0985 | 5 | 0.1196 | 1 | 0.389 | 2 |

FedAPEN's FM rank (3) and class-gap rank (5) are inverted. Across the five
runs, the two metrics still order the systems differently. This is an
observation, not a correlation claim: n=5 is insufficient to test whether the
reversal is systematic.

## Interpretation

Client-FM is a **1-D projection of a 2-D fairness object**. It answers “are
clients equal?” but marginalizes over “are classes equal?”. A method can
equalize clients while leaving a 44-point class spread untouched, or can have a
worse client-FM while being better balanced across classes. Reporting only FM
therefore hides a first-order property of the model.

## Proposed reporting metric

**Class-balanced FM** (lower is better):

```
class_FM = std_c( mean_i recall_{i,c} )
```

Report it alongside client-FM and **worst-class recall**. This is a
**reporting metric, not a training method**: it should not be optimized as an
objective, and it does not replace client-FM. The recommendation is to report
both axes and the worst-class tail.

## Negative result

**The participation hypothesis FAILED its pre-registered escalation** (FM
−2.84% relative; min-accuracy +4.09 points; rho difference 0.170), so C is
closed and is not resurrected here.

## What is missing

- **Second seed.** The half-scale results are one seed; class-std uncertainty
  is not estimated.
- **A larger run set.** Five runs are enough to observe order disagreement, not
  enough to test whether it is systematic.
- **FedAPEN entropy-correlated.** The FedAPEN run is linear only; the
  pattern-invariance claim remains a PHP-FL/Standalone observation.

## Caveats

- Half-scale CIFAR-10: 10 clients, 60 rounds, 1 seed, four CNNs, 1 local epoch,
  2,500 train / 500 test per client. This is ~8x less training than the paper;
  absolute AM/FM are not comparable to the paper.
- Class recalls are from the final-round per-client logs and the global
  ensemble recall file. One seed; no uncertainty estimate is claimed.
- The reconstructed smoke baselines (PHPFL/Standalone and B aggregation
  variants) are in `smoke_reconstructed_class_recall.csv`; they are a different
  scale and are not pooled with the half-scale numbers.
