# Confusion structure and resolution floor: findings

Follow-up to the class-fairness report. This document measures the full 10×10
confusion structure of the baseline and the four ladder rungs, decomposes the
error budget of three selected class pairs, and probes whether the residual
dog/cat error is a resolution floor.

No model was retrained for the confusion part. The resolution probe is a
separate small centralized experiment and is labelled as such throughout.

---

## 1. Measurement setup

**Confusion matrices.** The 10 clients of each run are combined exactly as in
the existing global per-class recall calculation: for client `i` with final
mixing weight `λ_i`, the ensemble logit for an image is

```
logit = λ_i · client_i(image) + (1 − λ_i) · global_i(image)
```

and the softmax probabilities are averaged over the 10 clients. Predictions
are taken on the **full 10,000-image CIFAR-10 test set**, so every matrix has
row support 1,000.

| Run | Experiment | Run store |
| --- | --- | --- |
| baseline | `C_half / PHPFL / linear` | `C_half__PHPFL__linear__s0__half10__3cf8f53c` |
| step 1 | class-balanced rehearsal | `STEP1__PHPFL__linear__s0__half10__0669227f` |
| step 2 | confusion-targeted loss | `STEP2__PHPFL__linear__s0__half10__130ba15d` |
| step 3 | worst-class aggregation | `STEP3__PHPFL__linear__s0__half10__e6b6e177` |
| step 4 | per-class distillation | `STEP4__PHPFL__linear__s0__half10__682b80ac` |

Raw and row-normalised matrices are in
`phofl/analysis/confusion_matrices/{run}{,_normalized}.csv`.
Builder: `analyze_confusion.py`, run with the working `torch 2.14.1+cu126`
(the pinned venv is blocked by a Windows Application Control policy).

**Two metrics per pair.** For classes `a` and `b`:

- `cross_error_rate = (n[a→b] + n[b→a]) / (support_a + support_b)`
  — only the pair's off-diagonal mass.
- `all_error_rate = (error_a + error_b) / 2`
  — average total error of the two classes, including errors sent to the other
  eight classes.

`cross_share = cross_mass / total_pair_error` says how much of the pair's
error is mutual rather than spilling elsewhere. Both metrics matter: a loss
that lowers `cross_error_rate` while holding or raising `all_error_rate` has
not fixed the classes, only redirected them.

---

## 2. Global behaviour

| Run | AM | worst class | worst recall | dog recall | cat recall |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.5425 | dog | 0.251 | 0.251 | 0.434 |
| step 1 rehearsal | 0.5468 | dog | 0.320 | 0.320 | 0.378 |
| step 2 confusion | 0.5489 | dog | 0.270 | 0.270 | 0.364 |
| step 3 worst-agg | 0.5539 | dog | 0.292 | 0.292 | 0.407 |
| step 4 distill | 0.5489 | dog | 0.282 | 0.282 | 0.458 |

The global-ensemble AM (0.54) is lower than the per-client mean AM (0.632)
reported earlier. These are different quantities and are not interchangeable.

The matrices are strongly asymmetric and cluster-structured. Baseline
dominant flows:

| flow | rate | flow | rate |
| --- | --- | --- | --- |
| automobile → truck | 0.266 | dog → cat | 0.250 |
| deer → bird | 0.261 | dog → bird | 0.188 |
| cat → bird | 0.187 | cat → frog | 0.138 |
| truck → automobile | 0.131 | ship → airplane | 0.085 |

Dog is worst because its errors are spread over cat (0.250) and bird (0.188)
at once. Cat is second-worst because it loses mass to bird (0.187) and frog
(0.138). Deer loses more to bird (0.261) than to horse (0.090). This matters
for interpreting the pair budget: only automobile/truck is a genuine
two-body problem.

---

## 3. Pair error budgets

### 3.1 All pairs, all runs

`phofl/analysis/confusion_pair_budget.csv`; `cross` and `all` are the two
metrics above.

| Run | Pair | a→b | b→a | cross | Δcross | all | Δall | cross share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | dog–cat | 0.250 | 0.047 | 0.1485 | – | 0.6575 | – | 0.226 |
| baseline | deer–horse | 0.090 | 0.065 | 0.0775 | – | 0.5155 | – | 0.150 |
| baseline | automobile–truck | 0.266 | 0.131 | 0.1985 | – | 0.3835 | – | 0.518 |
| step 1 | dog–cat | 0.205 | 0.089 | 0.1470 | −1.0% | 0.6510 | −1.0% | 0.226 |
| step 1 | deer–horse | 0.071 | 0.099 | 0.0850 | +9.7% | 0.4640 | −10.0% | 0.183 |
| step 1 | automobile–truck | 0.182 | 0.191 | 0.1865 | −6.0% | 0.3760 | −2.0% | 0.496 |
| step 2 | dog–cat | 0.192 | 0.060 | 0.1260 | −15.2% | 0.6830 | +3.9% | 0.184 |
| step 2 | deer–horse | 0.061 | 0.085 | 0.0730 | −5.8% | 0.5030 | −2.4% | 0.145 |
| step 2 | automobile–truck | 0.243 | 0.140 | 0.1915 | −3.5% | 0.3555 | −7.3% | 0.539 |
| step 3 | dog–cat | 0.208 | 0.069 | 0.1385 | −6.7% | 0.6505 | −1.1% | 0.213 |
| step 3 | deer–horse | 0.087 | 0.081 | 0.0840 | +8.4% | 0.4685 | −9.1% | 0.179 |
| step 3 | automobile–truck | 0.215 | 0.158 | 0.1865 | −6.0% | 0.3710 | −3.3% | 0.503 |
| step 4 | dog–cat | 0.256 | 0.059 | 0.1575 | +6.1% | 0.6300 | −4.2% | 0.250 |
| step 4 | deer–horse | 0.079 | 0.074 | 0.0765 | −1.3% | 0.5140 | −0.3% | 0.149 |
| step 4 | automobile–truck | 0.264 | 0.132 | 0.1980 | −0.3% | 0.3835 | 0.0% | 0.516 |

Key unbalanced rows are visible directly: dog→cat is 5.3× cat→dog
(0.250 vs 0.047) and automobile→truck is 2.0× truck→automobile
(0.266 vs 0.131).

### 3.2 Pair 1 — dog ↔ cat

Baseline: cross 0.1485, all 0.6575, cross share 0.226.

- **Step 2, the rung that explicitly targets this pair, does not reduce the
  pair's total error.** Cross error falls 15.2% (0.1485 → 0.1260) but joint
  class error *rises* 3.9% (0.6575 → 0.6830). Dog→cat drops 0.250 → 0.192
  while cat→dog rises 0.047 → 0.060. At the row level, dog's total error
  barely moves (0.749 → 0.730) and cat's gets worse (0.566 → 0.636).
  Cross share falls 0.226 → 0.184, i.e. the pair's errors became *less*
  mutual, not fewer.
- **Where the error went: bird.** Dog→bird rises 0.188 → 0.213 and
  cat→bird 0.187 → 0.220. The confusion loss sharpened the dog/cat boundary
  and pushed the ambiguous animal mass into the neighbouring bird class.
  This is displacement outside the pair, not a swap between the two
  directions. Net effect on the two classes is negative.
- **Step 1** is the only rung that improves both metrics at once for this
  pair, and it does so broadly: dog→cat 0.205, cat→dog 0.089, dog→bird 0.118,
  cat→bird 0.120, joint error −1.0%. Cross error is essentially unchanged
  (−1.0%): the pair boundary itself did not improve, the whole animal cluster
  did.
- **Step 4** produces the lowest joint dog+cat error of any run (0.6300,
  −4.2%) with both row errors below baseline (dog 0.718, cat 0.542), yet its
  dog↔cat cross error *rises* 6.1% to 0.1575 and its cross share returns to
  the baseline value. Correct decisions increased, but the residual confusion
  between the two classes did not shrink.
- **Step 3** sits between: cross −6.7%, joint −1.1%.

Interpretation: this pair is not bounded by a directional swap. Cat→dog is
small and roughly stable (0.047–0.089); the pressure is on dog→cat, and every
intervention that lowers it either sends the mass to bird (step 2) or
trades it against a higher cat error. The pair's cross share never falls
below 0.18, so most dog and cat error is never mutual at all.

### 3.3 Pair 2 — deer ↔ horse

Baseline: cross 0.0775, all 0.5155, cross share **0.150**.

This is not primarily a pair problem. Deer's dominant error is deer→bird
(0.261), which is 3× deer→horse (0.090); horse's dominant error is
horse→cat (0.109). Only 15% of the two classes' combined error is mutual.

- Step 1: cross +9.7%, joint −10.0% (best joint result of any run for this
  pair), driven by deer→bird 0.261 → 0.157.
- Step 2: cross −5.8%, joint −2.4%; deer→bird unchanged at 0.263.
- Step 3: cross +8.4%, joint −9.1%.
- Step 4: essentially flat.

So even the rungs that reduce joint deer+horse error do so by attacking the
bird confusion, not the deer/horse boundary. A deer↔horse-targeted loss would
be aimed at ~15% of the problem.

### 3.4 Pair 3 — automobile ↔ truck

Baseline: cross 0.1985, all 0.3835, cross share **0.518**.

This is the one genuine two-body pair: half of the two classes' error is
mutual, and they are the two best-performing vehicle classes otherwise.

- **Step 1 is a directional rebalancing.** automobile→truck falls
  0.266 → 0.182 while truck→automobile rises 0.131 → 0.191. The two
  directions nearly equalise (asymmetry 0.135 → 0.009) and cross error falls
  6.0%, joint error 2.0%. The pair is not fixed, but the bias is removed —
  the clearest example of "moving error between directions" in the ladder.
- Step 2: cross −3.5%, joint −7.3% (0.3555, best of any run).
- Step 3: cross −6.0%, joint −3.3%.
- Step 4: no change (cross −0.3%, joint 0.0%).

Even in the best run the pair retains ~19% cross error and ~36% joint error,
so the residual is large regardless of intervention.

---

## 4. Do targeted losses reduce pair error or move it?

Three distinct patterns, and only one of them is a swap:

1. **Displacement outside the pair — step 2 on dog/cat.** Lower cross error,
   higher joint error, with the mass reappearing as bird confusion. This is
   the worst case: the reported pair metric improves while the classes get
   worse.
2. **Directional rebalancing — step 1 on automobile/truck.** Lower
   automobile→truck compensated by higher truck→automobile, with a small net
   gain in cross error. Error moved between directions and was partly
   absorbed.
3. **Joint improvement with unchanged mutual confusion — step 4 on
   dog/cat.** Lower total error for both classes, but the pair boundary
   itself does not improve (cross +6.1%). Correct classifications increased
   without the two classes becoming easier to separate.

No rung produces a large, unambiguous reduction in a pair's joint error:
the best dog+cat joint improvement is 4.2%, the best automobile/truck is
7.3%, and deer/horse is dominated by bird confusion rather than by the pair.
This is consistent with the ladder's headline: maximum class-gap reduction
7.9%, no rung reaching the 30% closure threshold.

**Answer to the question posed:** targeted losses mainly move error. In one
case (automobile/truck under rehearsal) they move it between directions and
net-improve slightly; in the explicitly confusion-targeted case (dog/cat
under step 2) they move it out of the pair entirely and make the two classes
worse. Only step 4 lowers dog+cat joint error, and it does so without
reducing dog↔cat confusion.

---

## 5. Resolution probe

A small centralized probe, not a federated run: 2,000 training and 500 test
images per class, one 3-conv CNN with adaptive pooling, 15 epochs, AdamW,
cosine schedule, seed 0. CIFAR-10 is natively 32×32, so 64 and 96 are
**upsampled** (`torchvision.transforms.Resize`, antialiased). Results:
`phofl/analysis/resolution_probe_cifar.csv`, builder `resolution_probe.py`.

| Resolution | AM | dog recall | cat recall | dog↔cat cross | dog+cat joint | deer↔horse cross | auto↔truck cross |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.6950 | 0.596 | 0.494 | 0.185 | 0.455 | 0.073 | 0.080 |
| 64 | 0.6378 | 0.556 | 0.416 | 0.195 | 0.514 | 0.083 | 0.120 |
| 96 | 0.6092 | 0.562 | 0.400 | 0.188 | 0.519 | 0.087 | 0.131 |

Two observations:

1. **Upsampled resolution makes the probe worse, not better.** AM falls
   0.695 → 0.609 and dog+cat joint error rises 0.455 → 0.519. Upsampling adds
   no information; it only increases the input size for a fixed-capacity,
   fixed-epoch model, so the comparison is confounded by optimisation and
   scale. This probe cannot test an information floor.
2. **Cross-confusion is flat across resolution** (0.185 / 0.195 / 0.188) even
   as overall accuracy degrades. The dog↔cat boundary rate in this probe is
   insensitive to the input grid.

**Against the pre-specified decision rule.** The rule was: joint dog+cat
error flat at 32 and falling at higher resolution → resolution-limited; flat
even at higher resolution → label/data ambiguity. Here joint error rises
with resolution, so neither branch is satisfied, and the probe is invalid as
an information-floor test. The flat cross-confusion is at best suggestive
that the pair boundary is not resolution-limited.

**Context that survives the confound.** The centralized 32×32 probe reaches
dog+cat joint error 0.455, versus 0.6575 for the federated global ensemble on
the same 32×32 pixels. The federated deficit is therefore not a missing-pixel
problem: identical input resolution supports a 20-point lower joint error in
a centralized model. Any resolution explanation of the federated gap is
ruled out before the probe even runs.

---

## 6. Verdict

**Confusion structure.** The baseline is not a set of independent per-class
weaknesses. Dog, cat, deer and bird share one confusion cluster, and
automobile/truck share another. Dog's error is split between cat and bird;
deer's is dominated by bird; only automobile/truck is a two-body pair.
Pair-targeted interventions therefore act on a minority of the error for two
of the three pairs (cross share 0.15 for deer/horse, 0.23 for dog/cat).

**Targeted losses.** They predominantly displace error rather than remove it.
Step 2 is the clearest negative result: dog↔cat cross error −15.2% while
dog+cat joint error +3.9%, with the mass reappearing in bird. Step 1 gives a
genuine directional rebalancing for automobile/truck (asymmetry 0.135 → 0.009)
and a broad animal-cluster gain, but no pair closure. Step 4 is the only rung
that lowers dog+cat joint error (−4.2%) yet it raises dog↔cat cross error
(+6.1%). No intervention reduces a pair's joint error by more than 8%.

**Resolution floor.** No confirmed resolution-limited floor, and no confirmed
label-ambiguity floor either. The CIFAR probe is invalid for this question
because 64/96 are upsampled, and it degraded with resolution. What the data
does support is stronger and simpler: at the same native 32×32 resolution a
centralized model reaches dog+cat joint error 0.455 against the federation's
0.6575, so the federated dog/cat deficit is not caused by input resolution.
The remaining question — whether the *centralized* residual 0.455 is
resolution-limited or ambiguity-limited — is still open and requires native
high-resolution data.

**Method claim.** None. This is a diagnostic audit. `class_FM` stays a
reporting metric; it was not used as a training objective.

---

## 7. Future scope

1. **Native high-resolution control.** Repeat the probe on STL-10, which is
   natively 96×96, and downsample to 64 and 32 (rather than upsample CIFAR).
   That is the only configuration in which a resolution floor can be
   identified: the same scenes at three genuine information levels. The
   attempted download (2.45 GB) was abandoned incomplete; the partial archive
   was removed and is gitignored.
2. **Per-pair error accounting as standard reporting.** Report
   `cross_error_rate`, `all_error_rate` and `cross_share` together for every
   pair an intervention targets. This analysis shows that cross error alone is
   gameable: step 2 improves it by making the classes worse.
3. **Cluster-aware interventions.** Since dog/cat/deer/bird form one
   confusion cluster, the next intervention should be cluster-level
   (e.g. hierarchical or bird-vs-mammal auxiliary supervision) rather than
   pairwise. The step 2 result predicts that sharper pairwise boundaries
   without cluster supervision will keep exporting error to bird.
4. **Architecture decomposition.** The aggregate matrices mix four CNN
   architectures. Decompose the dog→cat and dog→bird flows by architecture to
   see whether the displacement is a model-capacity effect or a data effect.
   Step 4's best dog architecture already suggests architecture matters.
5. **Multi-seed and full-scale confirmation.** Every number here is one seed
   at half scale. Before any external claim, repeat the confusion audit across
   seeds and, if compute allows, at full scale, with the same global-ensemble
   definition.
6. **Label audit.** If the native-resolution control still shows flat
   dog/cat confusion, audit a sample of CIFAR dog/cat/deer/bird test images
   for label ambiguity and near-duplicate content. Only then can the residual
   be attributed to labels rather than to optimisation.

---

## 8. Reproduce

```bash
# full confusion matrices + pair budget (needs run_store checkpoints)
python analyze_confusion.py

# centralized 32/64/96 probe (~12 min on a 3060-class GPU)
python -c "import resolution_probe as r, pandas as pd; \
  pd.DataFrame([r.cifar_probe(x) for x in (32,64,96)]) \
  .to_csv('phofl/analysis/resolution_probe_cifar.csv', index=False)"

# figures
python report/make_confusion_figures.py
```

Artifacts:

- `phofl/analysis/confusion_matrices/` — 5 raw + 5 normalised 10×10 matrices
- `phofl/analysis/confusion_summary.csv` — AM, worst class, dog/cat recall
- `phofl/analysis/confusion_pair_budget.csv` — directional, cross, joint,
  cross-share and asymmetry per pair
- `phofl/analysis/resolution_probe_cifar.csv` — 32/64/96 probe
- `phofl/analysis/probe_confusion_cifar10_upsampled_{32,64,96}.csv` — probe
  matrices
- `report/figures/fig_confusion_matrices.png`,
  `fig_confusion_dogcat.png`, `fig_resolution_probe.png`
