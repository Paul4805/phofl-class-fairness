# C escalation report

A and B were not re-run.  No `C_findings.md` headline was written.

## 1. Standalone under entropy_correlated (existing CSV, no rerun)

| method | pattern | AM | FM | min | p10 | CVaR10 | rho(p, acc) |
|---|---|---:|---:|---:|---:|---:|---:|
| Standalone | entropy_correlated | 0.4605 | 0.185573 | 0.1000 | 0.1990 | 0.1000 | +0.5273 |

## 2. Holm-Bonferroni correction for the smoke C tests

| metric | comparison | raw p | Holm p | raw sig | Holm sig | loses sig |
|---|---|---:|---:|:---:|:---:|:---:|
| accuracy | PHPFL vs Standalone [linear] | 0.064453 | 0.193359 | no | no | no |
| accuracy | PHPFL vs Standalone [entropy] | 0.001953 | 0.009766 | yes | yes | no |
| accuracy | linear vs entropy [accuracy] | 0.410156 | 0.820312 | no | no | no |
| \|acc - group mean\| | linear vs entropy [dispersion] | 0.041016 | 0.164062 | yes | no | yes |
| PHPFL - Standalone | linear vs entropy [advantage] | 0.410156 | 0.820312 | no | no | no |

## 3. Pinned-stack smoke C

| pattern | method | AM | FM | min | p10 | CVaR10 | rho(p, acc) |
|---|---|---:|---:|---:|---:|---:|---:|
| linear | PHPFL | 0.5155 | 0.1511 | 0.2450 | 0.2990 | 0.2450 | -0.7212 |
| linear | Standalone | 0.4575 | 0.1826 | 0.1050 | 0.1950 | 0.1050 | -0.8424 |
| entropy_correlated | PHPFL | 0.5120 | 0.1744 | 0.2300 | 0.2570 | 0.2300 | +0.6930 |
| entropy_correlated | Standalone | 0.4575 | 0.1826 | 0.1050 | 0.1950 | 0.1050 | +0.6727 |

Sign check: FM entropy-linear = +0.0233, rho linear negative, rho entropy
positive.  Signs match the laptop smoke run.  Pinned dispersion raw p=0.083984
(vs laptop 0.041016), so significance did not replicate.

## 4. CIFAR-10 half-scale (final)

Configuration: 10 clients, 60 rounds, 1 seed, CIFAR-10, four CNNs (`HtM4`),
1 local epoch, 2,500 train / 500 test per client, pinned stack.

| method | pattern | AM | FM | min | p10 | CVaR10 | rho | worst class | worst recall |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| PHPFL | linear | 0.6320 | 0.0785 | 0.4920 | 0.5251 | 0.4920 | -0.1030 | dog | 0.2510 |
| PHPFL | entropy_correlated | 0.6308 | 0.0763 | 0.5329 | 0.5339 | 0.5329 | -0.2727 | dog | 0.2400 |
| Standalone | linear | 0.5429 | 0.0962 | 0.3683 | 0.4022 | 0.3683 | -0.1879 | cat | 0.2290 |
| Standalone | entropy_correlated | 0.5467 | 0.0985 | 0.3920 | 0.3985 | 0.3920 | -0.9152 | cat | 0.2180 |

Per-class recall:

| method | pattern | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PHPFL | linear | 0.609 | 0.571 | 0.610 | 0.434 | 0.382 | 0.251 | 0.694 | 0.588 | 0.623 | 0.662 |
| PHPFL | entropy_correlated | 0.626 | 0.613 | 0.599 | 0.457 | 0.397 | 0.240 | 0.673 | 0.511 | 0.616 | 0.696 |
| Standalone | linear | 0.405 | 0.586 | 0.564 | 0.229 | 0.335 | 0.285 | 0.615 | 0.381 | 0.488 | 0.541 |
| Standalone | entropy_correlated | 0.451 | 0.579 | 0.531 | 0.218 | 0.369 | 0.306 | 0.607 | 0.398 | 0.476 | 0.554 |

Paired tests (client-paired, raw and Holm):

| metric | comparison | test | statistic | raw p | Holm p | Holm sig |
|---|---|---|---:|---:|---:|:---:|
| accuracy | PHPFL vs Standalone [linear] | Wilcoxon | 0 | 0.001953 | 0.017578 | yes |
| accuracy | PHPFL vs Standalone [entropy] | Wilcoxon | 0 | 0.001953 | 0.017578 | yes |
| accuracy | linear vs entropy [accuracy] | Wilcoxon | 26.5 | 1.000000 | 1.000000 | no |
| dispersion | linear vs entropy [dispersion] | Wilcoxon | 27 | 1.000000 | 1.000000 | no |
| min-accuracy | linear vs entropy [min tail-shortfall] | Wilcoxon | 7 | 0.892738 | 1.000000 | no |
| CVaR10 | linear vs entropy [CVaR10 tail-shortfall] | Wilcoxon | 7 | 0.892738 | 1.000000 | no |
| min-accuracy | linear vs entropy [min] | paired bootstrap | -0.02557 | 0.208779 | 1.000000 | no |
| CVaR10 | linear vs entropy [CVaR10] | paired bootstrap | -0.02557 | 0.208779 | 1.000000 | no |
| advantage | linear vs entropy [advantage] | Wilcoxon | 26.5 | 1.000000 | 1.000000 | no |

Pre-registered verdict: **FAIL**.  FM increase is -2.84% relative (below the
10% and 16.96% STOP thresholds); min-accuracy improves by 4.09 points instead
of dropping; rho difference is 0.1697 (below the 1.0 GO threshold).

This is a compute-bounded half-scale, not the paper's 3-local-epoch setting;
absolute numbers are not comparable to the paper.
