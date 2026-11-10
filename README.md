# Class fairness in model-heterogeneous federated learning

[![CI](https://github.com/Paul4805/phofl-class-fairness/actions/workflows/ci.yml/badge.svg)](https://github.com/Paul4805/phofl-class-fairness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

This repository contains the measurement paper, analysis code, notebook, and
raw result tables for a two-axis fairness audit of model-heterogeneous
federated learning (MH-FL). It is **not** a new FL method.

## Finding

Client-FM (the standard deviation of per-client accuracy) hides a large class
axis. On the CIFAR-10 half-scale runs:

- PHP-FL/linear has a **44.3-point** best-to-worst class recall spread
  (frog 69.4%, dog 25.1%) while client-FM is 0.0785.
- The single best-client-FM run (PHP-FL/entropy) is worse on the class axis at
  45.6 points.
- FedAPEN/linear has a **52.9-point** gap, so the observation is field-wide
  across the two MH-FL methods tested (PHP-FL and FedAPEN), not universal.
- The two metrics rank the five runs differently; FedAPEN is 3rd by client-FM
  but 5th by class gap. This is an observation, not a correlation claim.

The diagnostic ladder reduced the gap by at most **7.9%** over one seed:

| rung | intervention | AM | FM | class gap | reduction | dog | frog |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | PHP-FL/linear baseline | 0.6320 | 0.0785 | 0.443 | — | 0.251 | 0.694 |
| 1 | class-balanced rehearsal | 0.6099 | 0.0854 | 0.408 | 7.9% | 0.320 | 0.728 |
| 2 | confusion-targeted loss | 0.6346 | 0.0900 | 0.450 | −1.6% | 0.270 | 0.720 |
| 3 | worst-class-weighted aggregation | 0.6312 | 0.1002 | 0.431 | 2.7% | 0.292 | 0.723 |
| 4 | per-class distillation | 0.6322 | 0.0944 | 0.424 | 4.3% | 0.282 | 0.706 |

Step 2 is the clearest negative signal: dog→cat errors fell 0.249→0.191, but
cat recall fell 0.434→0.364, and the combined dog+cat error rose
0.657→0.683. This is consistent with an information-theoretic floor at
CIFAR-10 resolution. The paper proposes `class_FM` as a **reporting metric**,
not a training objective.

## What is included

- `phofl/paper/` — abstract, introduction, related work, method, results,
  FedAPEN section, ladder section, discussion, conclusion.
- `phofl/analysis/` — 2-D client×class matrices, support-filtered analysis,
  AM-normalized analysis, ranking tables, and FedAPEN scope.
- `phofl_method/` — per-step ladder CSVs, findings, and `LADDER_SUMMARY.md`.
- `phofl_half/results/` — CIFAR-10 half-scale result CSVs.
- `phofl_half/logs/` — per-run round metrics and histories.
- `phofl/results/`, `phofl/logs/` — earlier A/B/C scoped runs.
- `phofl/figures/` — 300 DPI figures.
- `phofl_experiments.ipynb` — resumable A/B/C experiment notebook.
- `phofl_lab.py` — runtime harness used by the notebook and scripts.
- `analyze_two_axis.py` — rebuilds the 2-D analysis from logs.
- `run_step1.py` … `run_step4.py` — the diagnostic ladder.
- `run_c_half.py`, `run_fedapen_half.py`, `run_c_pinned_smoke.py` — C
  escalation and pinned-stack runs.
- `make_cifar10_10.py`, `cap_cifar10_10.py` — 10-client CIFAR data preparation.

## What is excluded

Large or reproducible artifacts are intentionally not committed:

- `third_party/` — clone the pinned PHP-FL repository locally.
- `phofl_half/data/`, `phofl_half/data_cap/` — generated CIFAR-10 partitions.
- `phofl/ckpts/`, `phofl_half/ckpts/`, `phofl_half/run_store/` — model
  checkpoints and scratch model files.
- `phofl_pinned/` — pinned-stack smoke working directory.
- `.venv-pinned/` — local Python environment.

`phofl/escalation/C_PREREG_COMMIT.txt` records the pre-registration commit hash.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

git clone https://github.com/Siyuan01/PHP-FL-main.git third_party/PHP-FL-main
git -C third_party/PHP-FL-main checkout 283c4a4c561b153d17ee0278ac35e5580b3fe28f
```

The notebook and scripts expect the repository at `third_party/PHP-FL-main`.
Data preparation for the half-scale CIFAR-10 setting:

```bash
python make_cifar10_10.py
python cap_cifar10_10.py
```

Rebuild the 2-D analysis from the committed result logs (no training):

```bash
python analyze_two_axis.py
```

Run the A/B/C notebook by opening `phofl_experiments.ipynb`; run the diagnostic
ladder with the `run_step*.py` scripts only if you intend to retrain.

## Caveats

- Half-scale: 10 clients, 60 rounds, one seed, four CNNs, one local epoch,
  2,500 train / 500 test per client. This is roughly 8× less training than the
  paper’s full CIFAR-10 setting; absolute AM/FM are not paper-comparable.
- The evidence is one seed. No seed-level uncertainty is claimed.
- Class cells with fewer than 10 test samples are excluded from the support-
  filtered analysis; the global class-level headline uses the full 1,000-sample
  CIFAR-10 test set per class.
- The participation hypothesis is closed and is not revived.
- No method contribution is claimed. `class_FM` is a reporting metric.

## Paper

The measurement paper is in `phofl/paper/`. It is deliberately not a method
paper; the diagnostic ladder is a negative result about reducibility.

## Paper and artifact map

- **Detailed findings PDF:** [`report/PHPFL_Class_Fairness_Report.pdf`](report/PHPFL_Class_Fairness_Report.pdf)
- **Results-only compendium:** [`report/PHPFL_All_Results.pdf`](report/PHPFL_All_Results.pdf)
- Paper source: `phofl/paper/` (abstract, introduction, related work, method,
  results, FedAPEN, diagnostic ladder, discussion, conclusion).
- Main finding: `phofl/FINDINGS_class_fairness.md`.
- Ladder negative result: `phofl_method/LADDER_SUMMARY.md`.
- Analysis tables: `phofl/analysis/`.
- Result CSVs and per-run logs: `phofl_half/results/`, `phofl_half/logs/`.
- Figures: `phofl/figures/`.
- Data/checkpoint policy: `DATA.md`.

## Citation

If you use this repository, please cite it using `CITATION.cff`:

```bibtex
@software{masillamni2026classfairness,
  author  = {Masillamni, J Paul},
  title   = {Class Fairness in Model-Heterogeneous Federated Learning:
             Measurement and a Diagnostic Ladder},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Paul4805/phofl-class-fairness},
  license = {MIT}
}
```

## Contributing

See `CONTRIBUTING.md`. This is a measurement repository: analysis,
reproducibility, and documentation contributions are welcome; new-method
claims are out of scope.

## Security

See `SECURITY.md`. Report vulnerabilities privately through GitHub Security
Advisories.

## License

Code and paper artifacts are released under the MIT License; see `LICENSE`.

## Acknowledgements

The experiments build on the PHP-FL and FedAPEN implementations and CIFAR-10.
See `ACKNOWLEDGEMENTS.md`; the external repositories are not vendored.

## Data availability

Generated CIFAR-10 partitions, checkpoints, scratch model stores, virtual
environments, and third-party clones are excluded from Git. `DATA.md` lists
what is excluded, why, and how to regenerate it. Result CSVs, structured logs,
and figures are included.
