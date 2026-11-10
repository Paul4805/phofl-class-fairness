# Class_FM diagnostic ladder

Same CIFAR-10 half-scale config as PHP-FL/linear: 10 clients, 60 rounds, one
seed, four CNNs, one local epoch, 2,500 train / 500 test per client.

| rung | intervention | AM | FM | class gap | gap reduction | dog recall | frog recall | closed? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | PHP-FL/linear baseline | 0.6320 | 0.0785 | 0.443 | — | 0.251 | 0.694 | — |
| 1 | class-balanced rehearsal (50/class, 1:1) | 0.6099 | 0.0854 | 0.408 | 7.9% | 0.320 | 0.728 | no |
| 2 | confusion-targeted loss (top-3, λ=0.1) | 0.6346 | 0.0900 | 0.450 | −1.6% | 0.270 | 0.720 | no |
| 3 | worst-class-weighted aggregation (γ=5.0) | 0.6312 | 0.1002 | 0.431 | 2.7% | 0.292 | 0.723 | no |
| 4 | per-class distillation (T=1.0) | 0.6322 | 0.0944 | 0.424 | 4.3% | 0.282 | 0.706 | no |

**Negative result.** No rung reached the pre-declared 30% class-gap reduction
threshold; the best was 7.9% from class-balanced rehearsal. Step 2 is the
clearest mechanism: dog→cat errors fell 0.249→0.191, but cat recall fell
0.434→0.364 and the combined dog+cat error rose 0.657→0.683. The 10×10
matrices show the displacement target: dog→bird 0.188→0.213 and cat→bird
0.187→0.220. The loss moved error out of the pair rather than removing it.
`class_FM` is a reporting metric, not an objective that a scalar penalty
reliably minimizes.

**No method contribution is claimed.**

A follow-up could test whether higher-resolution datasets show a reducible
class gap. Native high-resolution data is required: a centralized 32×32 probe
already reaches dog+cat joint error 0.455 against the federation's 0.657, so
the federated deficit is not an input-resolution effect and the upsampled
CIFAR 64/96 probe cannot test an information floor. See
`phofl/analysis/CONFUSION_RESOLUTION_FINDINGS.md`.
