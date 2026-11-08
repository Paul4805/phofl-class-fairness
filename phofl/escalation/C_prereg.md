# Pre-registration: C escalation to CIFAR-10 / 10 clients / 60 rounds

**Written before the 4-run half-scale table is available.**
Timestamp: 2026-11-07 (local)
Scope: only experiment C. A and B are closed and are not re-analysed here.

## Half-scale configuration being tested

- Dataset: CIFAR-10, 10 clients, Dirichlet(alpha=0.5), 2,500 train / 500 test
  per client (6 GB-feasible budget).
- 60 rounds, 1 seed, 4 CNNs (`HtM4`), 1 local epoch.
- Runs: PHP-FL and Standalone x linear and entropy_correlated (4 runs).
- This is **not** the paper's 3-local-epoch, full-data CIFAR-10 setting.
  Absolute AM/FM are not comparable to the paper. Only the linear-vs-entropy
  contrast is used.

## Endpoints (PHP-FL only)

Let `lin` and `ent` be the final per-client accuracy vectors.

- `FM_increase_rel = (FM_ent - FM_lin) / FM_lin`
- `min_drop_pts = 100 * (min(lin) - min(ent))`, in accuracy points.
- `rho_diff = |rho(ent) - rho(lin)|`, where `rho` is Spearman(p_i, accuracy).
- `effect_size` = `FM_increase_rel` (primary, directly comparable across
  scales). Secondary: paired dispersion Cohen's dz, reported but not gating.

Pinned-smoke reference (already observed, not a half-scale result):

- pinned smoke `FM_increase_rel` = (0.174359 - 0.151070) / 0.151070 = 15.42%
- pinned smoke `min_drop_pts` = 100 * (0.2450 - 0.2300) = 1.50 points
- pinned smoke `rho_diff` = |+0.6930 - (-0.7212)| = 1.4142

## Decision thresholds

### STOP if ANY of

1. `FM_increase_rel < 0.10` (less than 10% relative FM increase), or
2. `min_drop_pts < 2.0` (less than 2 accuracy points), or
3. `effect_size <= 1.10 * pinned_smoke_FM_increase_rel` (i.e. <= 16.96%).

### GO if ALL of

1. `FM_increase_rel > 0.25` (>25% relative FM increase), and
2. `min_drop_pts > 5.0` (>5 accuracy points), and
3. `rho_diff > 1.0`, and
4. only then: propose 2 additional seeds at half scale.

### MARGINAL if between

Do not escalate further. Report as an exploratory result with the explicit
caveat that it did not survive the pre-registered escalation thresholds.

## Statistical reporting

- Unit: client, paired by `client_id`. One seed.
- Raw and Holm-Bonferroni p-values for the half-scale C family.
- Wilcoxon on per-client accuracy and on per-client **tail shortfall**
  `max(0, group_mean - accuracy)` as the paired tail-sensitive test.
- Because min and CVaR10 are group-level scalars, the direct tests for their
  differences use a paired client bootstrap (10,000 resamples); the
  tail-shortfall Wilcoxon is reported alongside. With 10 clients, the lowest
  10% is one client, so CVaR10 equals min; both are still reported.
- No wording such as "confirms" or "supports" the smoke result. The half-scale
  run only tests these pre-registered thresholds.
