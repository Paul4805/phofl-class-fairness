# Data and artifact availability

This repository keeps the source, result CSVs, structured per-run logs, and
figures. It does not keep generated datasets, model checkpoints, or scratch
model stores.

## Excluded from Git

| Path | Size (approx.) | Why excluded | How to regenerate |
|---|---:|---|---|
| `third_party/` | 0.6 GB | external repositories | clone PHP-FL at the pinned SHA (README) |
| `phofl_half/data/` | 0.2 GB | generated CIFAR-10 partitions | `python make_cifar10_10.py` |
| `phofl_half/data_cap/` | 0.1 GB | capped CIFAR-10 partitions | `python cap_cifar10_10.py` |
| `phofl_half/run_store/` | 4.7 GB | per-client scratch model files | recreated by training scripts |
| `phofl_half/ckpts/` | 2.7 GB | resumable checkpoints | recreated by training scripts |
| `phofl/ckpts/` | 2.0 GB | resumable checkpoints | recreated by training scripts |
| `phofl_pinned/` | 2.6 GB | pinned-stack smoke workspace | recreated by `run_c_pinned_smoke.py` |
| `.venv-pinned/` | 5.0 GB | local environment | `requirements.txt` |

## Included result artifacts

- `phofl/analysis/` — client × class matrices, support filtering, AM
  normalization, ranking disagreement, FedAPEN scope.
- `phofl_half/results/` — half-scale result CSVs, including per-class recall.
- `phofl_half/logs/` — per-run `round_metrics.csv`, histories, and metadata.
- `phofl/results/` and `phofl/logs/` — earlier A/B/C scoped runs.
- `phofl_method/` — diagnostic-ladder per-step CSVs and findings.
- `phofl/figures/` — 300 DPI figures.

## Reproducing the analysis without training

```bash
python analyze_two_axis.py
python make_final_table.py
```

These commands read the committed logs/results and do not fit models.
