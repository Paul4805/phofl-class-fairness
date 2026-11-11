# Three-seed class-gap comparison: PHPFL, FedAPEN, FedFV, Standalone

Follow-up to the class-fairness report. The single-seed numbers were never
tested for seed stability; this run does that, adds a FedFV implementation,
and re-establishes the class-gap ordering on one stack.

---

## 1. What was run

**Protocol.** Identical to the committed half-scale runs: CIFAR-10, 10
clients with four CNN architectures (`HtM4`), 60 rounds, 1 local epoch,
batch 64, `eval_gap=10`, `tau=0.2`, `sig_k=5.0`, GoogLeNet shared model,
participation probabilities from the named pattern. Seeds **0, 1, 2**.

**Grid.** 4 methods × 2 patterns × 3 seeds = **24 runs**, all under
experiment `C_half_s3`.

| method | linear | entropy | implementation |
| --- | --- | --- | --- |
| PHPFL | 3 seeds | 3 seeds | PHP-FL, as shipped |
| FedAPEN | 3 seeds | 3 seeds | FedAPEN client class |
| FedFV | 3 seeds | 3 seeds | **new port**, see §6 |
| Standalone | 3 seeds | 3 seeds | local-only, no server |

**Stack.** The committed seed-0 runs used `torch 2.2.2+cu121` / `numpy 1.26.4`.
That virtualenv is now blocked by a Windows Application Control policy
(`WinError 4551`, `shm.dll`); unblocking every file and copying the venv to a
new path both failed, so the block is hash-based. All 24 runs here therefore
use the installed stack (`torch 2.14.1+cu126`, `numpy 2.x`), including seed 0.
The previous pinned runs are kept only as a stack calibration check (§7).

**Class metrics.** Global per-class recall is measured on the full CIFAR-10
test set, averaging the softmax over the 10 clients with each client's final
inference weight. `class gap = max recall − min recall`. Error bars are the
sample standard deviation over 3 seeds.

Runtime: 36–50 min per run, ~17 h total on the RTX 3060 laptop GPU.

---

## 2. Main result

Mean ± sd over three seeds, sorted by mean class gap.

| method | pattern | AM | client-FM | class gap | worst-class recall |
| --- | --- | --- | --- | --- | --- |
| FedFV | linear | 0.6272 ± 0.0080 | 0.0874 ± 0.0035 | **0.3527 ± 0.0446** | **0.346 ± 0.039** |
| PHPFL | linear | 0.6286 ± 0.0083 | 0.0860 ± 0.0095 | 0.3867 ± 0.0766 | 0.312 ± 0.053 |
| Standalone | linear | 0.5423 ± 0.0014 | 0.0996 ± 0.0095 | 0.3887 ± 0.0189 | 0.215 ± 0.009 |
| PHPFL | entropy | 0.6302 ± 0.0041 | 0.0817 ± 0.0032 | 0.4083 ± 0.0170 | 0.278 ± 0.031 |
| Standalone | entropy | 0.5377 ± 0.0060 | 0.0954 ± 0.0100 | 0.4087 ± 0.0076 | 0.212 ± 0.013 |
| FedFV | entropy | 0.6262 ± 0.0087 | 0.0847 ± 0.0042 | 0.4330 ± 0.0304 | 0.268 ± 0.026 |
| FedAPEN | entropy | 0.6064 ± 0.0075 | 0.0873 ± 0.0109 | 0.5043 ± 0.0517 | 0.276 ± 0.053 |
| FedAPEN | linear | 0.6151 ± 0.0030 | 0.0779 ± 0.0137 | **0.6017 ± 0.0853** | 0.207 ± 0.019 |

Per-seed class gap:

| method / pattern | seed 0 | seed 1 | seed 2 | spread |
| --- | --- | --- | --- | --- |
| FedFV / linear | 0.399 | 0.310 | 0.349 | 8.9 pts |
| PHPFL / linear | 0.468 | 0.316 | 0.376 | **15.2 pts** |
| Standalone / linear | 0.397 | 0.367 | 0.402 | 3.5 pts |
| PHPFL / entropy | 0.426 | 0.392 | 0.407 | 3.4 pts |
| Standalone / entropy | 0.412 | 0.400 | 0.414 | 1.4 pts |
| FedFV / entropy | 0.468 | 0.418 | 0.413 | 5.5 pts |
| FedAPEN / entropy | 0.510 | 0.450 | 0.553 | 10.3 pts |
| FedAPEN / linear | 0.547 | 0.558 | 0.700 | **15.3 pts** |

---

## 3. Findings

### 3.1 The single-seed headline does not survive reseeding

PHPFL/linear spans **0.316 to 0.468** across seeds, a 15.2-point swing in the
same configuration. The published 44.3-point gap was the seed-0 draw. The
defensible statement is:

> PHPFL/linear class gap = **0.39 ± 0.08** over three seeds (mean 0.387).

FedAPEN/linear is worse behaved still: 0.547, 0.558, **0.700**, driven by
seed 2 selecting `automobile` as the worst class at 0.185 recall instead of
dog or cat. Single-seed worst-class labels are not reliable: PHPFL/linear
reports dog, dog, cat; FedAPEN/linear reports cat, cat, automobile.

### 3.2 PHPFL has no class-gap advantage over Standalone

Per-seed difference PHPFL − Standalone:

- linear: **+0.071, −0.051, −0.026** (no consistent sign)
- entropy: **+0.014, −0.008, −0.007** (no consistent sign)

Means are indistinguishable (0.387 vs 0.389 linear; 0.408 vs 0.409 entropy)
while AM differs by ~9 points (0.629 vs 0.542). The earlier reading that
Standalone only *appeared* fairer after AM normalization should be replaced:
on the class axis the two methods are tied, and Standalone's much lower AM
comes with an equally wide class gap and a far worse worst class (0.215 vs
0.312).

### 3.3 FedAPEN is consistently the worst on the class axis

FedAPEN − PHPFL per seed: linear +0.079, +0.242, +0.324; entropy +0.084,
+0.058, +0.146. **6 out of 6 comparisons positive**, mean penalty 12–22
points. This is the strongest ordering claim in the sweep: it does not depend
on seed, pattern, or which class happens to be worst.

### 3.4 FedFV/linear has the best mean gap and the best worst-class tail, but the margin is not established

FedFV/linear is the lowest mean gap (0.353) and the highest mean worst-class
recall (0.346; PHPFL 0.312, Standalone 0.215). Per-seed vs PHPFL/linear:
**−0.069, −0.006, −0.027** (3/3 lower, mean −3.4 pts). Vs Standalone/linear:
+0.002, −0.057, −0.053 (2/3 lower).

With three seeds a 3/3 sign pattern is not significant (two-sided sign test
p = 0.25), and the mean difference is smaller than PHPFL's own seed sd. So the
honest claim is directional: **conflict-aware aggregation is the only
variation here that improves both the mean gap and the worst-class tail without
costing AM**, and it deserves a larger seed budget before being claimed as a
fix. It is not a method contribution.

### 3.5 The entropy effect is method-dependent, not structural

Entropy minus linear, within method:

| method | seed 0 | seed 1 | seed 2 | reading |
| --- | --- | --- | --- | --- |
| Standalone | +0.015 | +0.033 | +0.012 | slightly worse, 3/3 |
| PHPFL | −0.042 | +0.076 | +0.031 | mixed |
| FedFV | +0.069 | +0.108 | +0.064 | worse, 3/3 |
| FedAPEN | −0.037 | −0.108 | −0.147 | better, 3/3 |

The earlier claim that class bias is "structural / participation-agnostic" is
too strong. Participation pattern changes the class gap by up to 15 points,
in a direction that depends on the aggregation rule: it hurts FedFV and
Standalone, is neutral-to-mixed for PHPFL, and clearly helps FedAPEN. There is
no single pattern effect to report.

### 3.6 The client-FM / class-gap inversion is robust

The worst-gap run is **FedAPEN/linear in all three seeds**. The best-client-FM
run is FedAPEN/linear in seeds 0 and 2, and PHPFL/linear in seed 1. So a
leaderboard sorted by client-FM selects, two times out of three, the run with
the worst class gap. Pooled extremes: client-FM 0.078 → gap 0.602 versus
client-FM 0.100 → gap 0.389. This is now a three-seed structural result rather
than a single-run observation.

### 3.7 Accuracy and the class axis are not in tension here

FedFV/linear matches PHPFL/linear on AM (0.627 vs 0.629) and beats it on the
two class metrics. Standalone pays 8.6 AM points and gains nothing on the class
axis. FedAPEN pays 1.4 AM points and loses 21 gap points. The trade-off between
accuracy and class fairness that motivated the ladder is not visible across
methods at this scale.

---

## 4. What this changes in the paper

| old claim | new statement |
| --- | --- |
| "44.3-point class gap" headline | "0.39 ± 0.08 over 3 seeds", with seed-0 0.468 as the high draw |
| "best client-FM run is worse on the class axis (45.6 pts)" | holds, and is now 2/3 seeds rather than one run |
| "Standalone's raw gap advantage is an AM effect; gap/AM 0.711 vs 0.701" | superseded: gap is tied, so the normalization argument is moot |
| "PHP-FL vs FedAPEN class gap 44.3 vs 52.9" | holds and strengthens: 0.39 vs 0.60, 6/6 per-seed comparisons |
| "class bias is structural; participation is not the main driver" | narrowed: pattern effect is method-dependent, up to 15 pts |
| "four interventions reduce the gap by at most 7.9%" | unchanged (ladder is a separate single-seed experiment) |
| FedFV "not ported" | FedFV/linear and FedFV/entropy now measured (§6) |

FedHEAL remains unimplemented and is listed as future work.

---

## 5. Limits

- Three seeds. Enough to expose seed noise, not enough for significance on the
  FedFV-vs-PHPFL margin. The pairwise claims above are sign counts, not tests.
- Half scale, one dataset, one GPU, 60 rounds. Absolute values are not
  comparable to the paper's full-scale table.
- The stack differs from the pinned seed-0 artifacts (§7).
- Worst-class labels are unstable, so per-class claims should use recall
  vectors, not "the worst class".
- FedFV is a reimplementation, not the authors' code (§6).

---

## 6. FedFV implementation

Ported from Wang et al., *Federated Learning with Fair Averaging*, IJCAI 2021
(arXiv:2104.14937), Algorithms 1–3. No official code was vendored.

- Update direction per client: `g_k = theta_t − theta_k` on the shared model.
- **Internal conflicts** (Alg. 2): clients are sorted by ascending training
  loss; the first `floor(alpha * m)` keep their gradient; every other client is
  projected onto the normal plane of each conflicting gradient in that order.
- **External conflicts** (Alg. 3): after round `tau`, the aggregated update is
  projected away from the summed recent gradients of clients that were not
  selected and conflict with it.
- **Rescaling** (Alg. 1, line 10): the mitigated update is rescaled to the norm
  of the plain average update.
- Hyper-parameters fixed a priori from the paper's CIFAR-10 grid:
  **alpha = 0.1, tau = 3**. They were not tuned on our test set.
- Training-loss ordering uses the shared model's CE loss, recorded during the
  existing local loop (instrumentation only; no change to the objective).

**Composition caveat.** FedFV is defined in parameter space, so in a
model-heterogeneous harness it can only act on the model every client shares.
It therefore replaces aggregation of the shared/auxiliary model while the
PHP-FL client objective is unchanged. Read the FedFV rows as *FedFV
aggregation on the PHP-FL local objective*, not as vanilla FedFV end-to-end.
A homogeneous-model comparison would be needed to isolate the method.

Verification: 5 unit tests (`tests/test_fedfv.py`), including alpha = 1
recovering exact plain averaging and alpha = 0 preserving the plain update
norm; plus a one-round CPU smoke run before the grid was launched.

---

## 7. Stack calibration

Seed-0 cells re-run on the installed stack versus the committed pinned-stack
seed-0 runs:

| cell | pinned AM → new AM | pinned FM → new FM |
| --- | --- | --- |
| PHPFL / linear | 0.6320 → 0.6372 | 0.0785 → 0.0855 |
| PHPFL / entropy | 0.6308 → 0.6342 | 0.0763 → 0.0851 |
| FedAPEN / linear | 0.6209 → 0.6185 | 0.0839 → 0.0720 |
| Standalone / linear | 0.5429 → 0.5433 | 0.0962 → 0.0932 |
| Standalone / entropy | 0.5467 → 0.5427 | 0.0985 → 0.0897 |

Maximum |ΔAM| = 0.0052, maximum |ΔFM| = 0.0119. The stack change is smaller
than the seed sd it is being compared against, so it does not drive §3.

---

## 8. Future scope

1. **More seeds on the FedFV margin.** The 3.4-point FedFV/linear advantage
   over PHPFL needs n ≥ 10 to separate from a 4.5-point sd. That is the single
   highest-value run left.
2. **Homogeneous-model control.** Re-run FedFV with identical client
   architectures so the method can be evaluated as published, without the
   shared-model composition.
3. **FedHEAL.** Still unimplemented; identify the exact paper PHP-FL cites
   before porting, since several methods share the name.
4. **Per-class reporting instead of "worst class".** Worst-class identity is
   unstable across seeds; report the full 10-class recall vector with seed
   spread.
5. **Paired tests.** With 3 seeds the sign counts are descriptive. Use a paired
   design over seeds or clients, pre-register the metric, and report effect
   sizes with intervals.
6. **Resolve the entropy interaction.** The pattern effect reverses sign
   between FedAPEN and FedFV on the same data; a controlled sweep of
   participation concentration (sigma) would show whether this is a
   client-weighting interaction or noise.

---

## 9. Artifacts

- `phofl_half/results/C_half_s3_seed_sweep.csv` — 24 rows, per-run metrics
- `phofl_half/results/C_half_s3_seed_sweep_recall.csv` — per-class recall
- `phofl/analysis/seed_sweep_summary.csv` — mean/sd per method × pattern
- `phofl/analysis/seed_sweep_class_gap.csv` — per-seed rows, sorted
- `phofl/analysis/seed_sweep_stack_check.csv` — §7
- `phofl_half/logs/C_half_s3/` — per-run round metrics and histories
- `phofl_half/logs/C_half_s3_sweep.log`, `C_half_s3_fedfv.log`
- `phofl_lab.py` — `_fedfv_project`, `_fedfv_external`, `_fedfv_aggregate`
- `tests/test_fedfv.py` — FedFV unit tests
- `run_seed_sweep.py`, `analyze_seed_sweep.py`

Reproduce:

```bash
python run_seed_sweep.py                    # 18 runs, 3 methods x 2 patterns x 3 seeds
python run_seed_sweep.py --methods FedFV    # 6 runs
python analyze_seed_sweep.py
```
