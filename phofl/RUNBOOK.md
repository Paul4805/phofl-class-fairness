# RUNBOOK - scoped PHP-FL experiments

## What is in scope

- **A** fairness-metric recomputation from the plain baseline logs (0 compute
  once B/plain exists).
- **B** aggregation reweighting: `plain`, `inv_p`, `inv_N`, PHP-FL only,
  10 clients, 40 rounds, Fashion-MNIST, seed 0.
- **C** correlated participation: `linear`, `entropy_correlated` x
  PHP-FL, Standalone, same smoke scale.

E4-E8 are intentionally absent.  The full 170-hour matrix was not launched.

## Resume

1. Re-run Cells 0-2 (install, Drive paths, repo clone + patches).
2. Re-run the experiment cell.  It checks `results/{exp}/{run_id}_complete.flag`
   and the latest `ckpts/{run_id}/..._r*.pt`.
3. A run that is already complete is skipped.  A partial run resumes after the
   latest checkpoint (every 10 rounds).

## Scale / cost

Smoke is 10 clients x 40 rounds x 1 seed.  A complete B+C pass is six training
runs: B (3 aggregation modes) + C (2 PHP-FL patterns + 1 shared Standalone).
The Standalone run is reused for both C patterns because it has no server.

If you escalate C to CIFAR-10 / half-scale, restore `model_family="HtM4"` and
the 4-CNN round-robin, use 20 clients/60 rounds/2 seeds, and set `SCOPE` in
`phofl_lab.py` accordingly.

## Artifacts

- `results/A_fairness.csv`, `B_results.csv`, `C_results.csv`, `STATS_tests.csv`
- `logs/{exp}/{run_id}/round_metrics.csv`, `round_log.csv`, `G_eval.csv`
- `ckpts/{run_id}/..._r*.pt` (latest checkpoint per run)
- `figures/*.png` (300 DPI)
- `RESULTS.md`, `STATS.md`, `VERIFY.md`, `RUNBOOK.md`

## Failure policy

Every run writes a traceback to `failures.log`, preserves the latest
checkpoint, and refuses to mark itself complete.  No exception is swallowed.

## Approval gate

The original 170-hour plan stopped after a CIFAR-10 baseline for human
approval.  This scope-change run is the approved reduced matrix, so it proceeds
directly through A, B and C; no separate `APPROVE_BASELINE` sentinel is used.
