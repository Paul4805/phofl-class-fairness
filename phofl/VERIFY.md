# VERIFY - patches, data, and non-obvious choices

## Pinned repository

- `https://github.com/Siyuan01/PHP-FL-main.git`
- commit `283c4a4c561b153d17ee0278ac35e5580b3fe28f`
- working copy: `C:\Users\J Paul Masillamni\Documents\projects\fl\third_party\PHP-FL-main`

## Patches applied (runtime, idempotent)

1. **Aggregation hook** (`PHPFL.aggregate_parameters`): the repository's
   `PHPFL` override is a plain mean over active clients. We replace only the
   weight computation with `aggregation_weights(server, mode)`:
   `plain` = 1/|A|, `inv_p` = 1/p_i, `inv_N` = 1/max(N_i,1),
   `fedavg` = n_i. Weights are normalized. `plain` is the untouched baseline.
2. **AMP client training** (`clientPHP.train`, `learn_weight_for_inference`):
   the same DEAL losses and optimizer updates are wrapped in
   `torch.cuda.amp.autocast(dtype=torch.float16)` + `GradScaler`.
3. **CPU-backed model zoo**: `save_item` writes CPU models and `load_item`
   moves one model to the active device. This keeps the 6 GB card from holding
   20 w_i/g_i pairs. `filter_top_p_percent` and `update_model_with_mask` move
   one mask at a time.
4. **ISPU masks** stay on CPU after each update; they are recomputed every
   participation and are not part of the persistent GPU state.
5. **Logging cost control**: `round_log.csv` records every (round, client) row,
   but the per-active-client local/aux/ensemble evaluation is disabled in the
   scoped runs (`log_active_metrics=False`) to keep the 6 GB smoke run under
   three hours.  Full per-client accuracy vectors are in
   `round_metrics.csv` every 5 rounds for all 10 clients; A/B/C read those.

## Data

- CIFAR-10 is generated with the repository's `dataset/generate_cifar10.py
  noniid - dir` (Dirichlet alpha=0.5, 20 clients, 15% test holdout). The
  generated label statistics were verified byte-for-byte against the
  repository's shipped `config.json`.
- The scope-change runs are Fashion-MNIST, 10 clients, Dirichlet alpha=0.5.
  Because the paper code does not ship Fashion-MNIST, a shim writes the same
  `.npz` layout. Smoke caps are 1000 train / 200 test per client, so 10 clients
  x 40 rounds fit a 6 GB laptop GPU. This is a smoke diagnostic, not a claim
  about full-scale CIFAR-10.
- Smoke uses the two fastest official backbones (GoogLeNet, ResNet18) instead
  of the paper's four CNNs. Escalation to CIFAR-10/half-scale restores HtM4.

## Scope-change decisions

- **Entropy-correlated participation**: the original prompt gave
  `p_i = exp(-H(D_i))` but described high-entropy clients as participating
  more. Those are opposite directions. We use the normalized entropy
  `p_i = clip(exp(H - ln K), 0.02, 1)` so that balanced clients participate
  more and rare-class holders less, matching the stated intent.
- **Standalone** is server-free and therefore pattern-invariant. It is run
  once and reused for the linear and entropy rows of C; this is explicit in
  `C_results.csv`.
- **Wilcoxon unit**: clients, paired by `client_id`. Seeds are not replicates
  (one seed in the smoke scope). With 10 clients the minimum two-sided
  p-value is 0.00195.
- **E4-E8** are not run in this scope.

## Environment

The notebook installs `torch==2.2.2+cu121`, `numpy==1.26.4` and the pinned
Colab stack.  The local execution used to produce these artifacts ran on the
installed laptop stack shown below (torch 2.14.1 / CUDA 12.6 / numpy 2.5.4).
That is a deliberate 6 GB fallback, not the paper's pinned environment; re-run
the smoke scope on the Colab pins before treating the numbers as final.

```json
{
  "python": "3.12.10",
  "torch": "2.14.1+cu126",
  "cuda": "12.6",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "gpu_mb": "6144",
  "numpy": "2.5.4",
  "pandas": "3.1.1",
  "scipy": "1.18.1",
  "sklearn": "1.9.1",
  "matplotlib": "3.11.2",
  "seaborn": "0.13.2",
  "PIL": "13.0.0",
  "tqdm": "4.70.1",
  "thop": "0.1.1"
}
```

## C escalation / half-scale note

The CIFAR-10 C escalation uses **1 local epoch, 60 rounds, 10 clients, and
2,500 train / 500 test samples per client** (four CNNs, one seed).  That is
roughly **8x less total client training** than the paper's full CIFAR-10
setting.  The absolute AM/FM numbers from this run are **NOT comparable** to
the paper's Table 2 or to the full-scale CIFAR-10 results; only the
linear-vs-entropy contrast is usable.  The run tests the thresholds in
`phofl/escalation/C_prereg.md` (commit `9e54d472fd87b44a4014219507f43cc217ca53fd`)
and must not be described as confirming or supporting the smoke result.

## Class-fairness analysis follow-up

The 2-D client-by-class analysis is analysis-only; no training was run for it.
Per-client class recalls are available for the five half-scale runs (including
FedAPEN) and for the earlier B/C smoke runs, and were reconstructed into
`phofl/analysis/smoke_reconstructed_class_recall.csv`.

**FedAPEN completed.** The queued FedAPEN/linear half-scale run finished
without OOM at batch 64 (41.2 min, local RTX 3060). It has client-FM 0.0839,
class gap 52.9 points, support-filtered gap 41.6 points, and ranks 3rd by
client-FM but 5th by class gap. The >30-point pre-registered threshold promotes
the claim to field-wide across the two MH-FL methods tested (PHP-FL and
FedAPEN); it is not a universal FL claim.

**Support analysis.** Per-(client,class) test counts were loaded from
`phofl_half/data_cap/Cifar10/test/*.npz`.  Each run has 34/100 cells with
fewer than 10 samples (170 total across the five runs, including 45
zero-sample cells), excluded from the worst-cell calculation and listed in
`phofl/analysis/low_support_cells.csv`.  The worst supported cell is 0.0
(client 0, cat, 11 samples) in every run; it is sparse but not below the
threshold.  The class-level 44.3-point spread is based on the full
1,000-sample CIFAR-10 test set per class and is not affected by the cell
support filter.

Support sensitivity: the global class-level 44.3-point spread is conservative.
Restricting class means to client-cells with at least 10 test samples raises
the PHP-FL/linear spread to 51.3 points (worst class deer), the entropy spread
to 45.0 points, and the FedAPEN spread to 41.6 points.  The 34/100 low-support
cells per run are in `phofl/analysis/low_support_cells.csv`.
