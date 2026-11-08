# 4b. FedAPEN: field-wide scope test

## Configuration

- Method: FedAPEN
- Participation: linear
- Seed: 0 (one seed; matches the existing four runs)
- Data: CIFAR-10, 10 clients, 2,500 train / 500 test per client
- Training: 60 rounds, four CNNs (`HtM4`), 1 local epoch
- Stack: pinned `torch==2.2.2+cu121`, `numpy==1.26.4`
- Same per-class recall logging as the four completed runs
- Completed without OOM at batch 64; no retry was needed.

## Result

| metric | FedAPEN / linear |
|---|---:|
| client-FM | 0.0839 |
| class-std | 0.1698 |
| worst class | cat |
| worst-class recall | 0.256 |
| best class | frog |
| best-class recall | 0.785 |
| class gap | **0.529** |
| support-filtered class gap | 0.416 |
| client-FM rank (5 runs) | 3 |
| class-gap rank (5 runs) | 5 |

Per-class recall:

| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.520 | 0.704 | 0.631 | 0.256 | 0.482 | 0.275 | 0.785 | 0.639 | 0.663 | 0.424 |

## Conditional scope

The pre-specified rule was:

- class gap > 30 points → promote to field-wide;
- class gap < 20 points → narrow to PHP-FL;
- 20–30 points → mixed.

FedAPEN's class gap is **52.9 points**, above the 30-point threshold. The claim
scope is therefore promoted, but only to the two MH-FL methods tested:

> Across PHP-FL and FedAPEN, client-FM can hide a >40-point class-recall
> spread.

This is not a universal claim about all FL methods. It is a field-wide claim
over the two MH-FL methods in this study, based on the pre-specified rule. The
FedAPEN rank by client-FM (3rd) and by class gap (5th, worst) are inverted.
