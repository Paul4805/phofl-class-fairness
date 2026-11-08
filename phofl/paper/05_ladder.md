# 5. Can class_FM be reduced? A diagnostic ladder

The class-FM finding would be less interesting if a standard intervention
removed it. We therefore ran a four-rung diagnostic ladder on the same
CIFAR-10 half-scale configuration as the baseline PHP-FL/linear run: 10
clients, 60 rounds, one seed, four CNNs, one local epoch, 2,500 train / 500
test per client. Each rung changed exactly one mechanism. The baseline is the
existing PHP-FL/linear run.

| rung | intervention | AM | FM | class gap | gap reduction | dog recall | frog recall | closed? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | PHP-FL/linear baseline | 0.6320 | 0.0785 | 0.443 | — | 0.251 | 0.694 | — |
| 1 | class-balanced rehearsal, 50 samples/class, 1:1 mix | 0.6099 | 0.0854 | 0.408 | 7.9% | 0.320 | 0.728 | no |
| 2 | confusion-targeted loss, top-3 pairs, λ=0.1 | 0.6346 | 0.0900 | 0.450 | −1.6% | 0.270 | 0.720 | no |
| 3 | worst-class-weighted aggregation, γ=5.0 | 0.6312 | 0.1002 | 0.431 | 2.7% | 0.292 | 0.723 | no |
| 4 | per-class distillation, T=1.0 | 0.6322 | 0.0944 | 0.424 | 4.3% | 0.282 | 0.706 | no |

No rung reached the pre-declared 30% gap-reduction threshold. The largest
reduction was **7.9%**, from class-balanced rehearsal.

## 5.1 Class-balanced rehearsal: broad improvement, preserved gap

Step 1 improved both ends of the class distribution: dog recall rose from
0.251 to 0.320 (+6.9 points) and frog recall from 0.694 to 0.728 (+3.4
points). The absolute best-to-worst gap fell only from 0.443 to 0.408, and the
normalized gap barely moved (gap/AM 0.701 → 0.669). Both classes improved, but
the improvement was proportional rather than redistributive. The data-local
buffer helps the rare class without changing the class geometry enough to
close the gap.

## 5.2 Confusion-targeted loss: a zero-sum trade

Step 2 added a margin penalty on each client's top-3 validation confusion
pairs (λ=0.1, margin 1.0). The targeted pair moved in the intended direction:
the dog→cat error rate fell from **0.249 to 0.191** (−23%). But cat recall fell
from **0.434 to 0.364** (−16%), and the combined dog+cat error rate rose from
**0.657 to 0.683**:

```
baseline: (1 - 0.251 + 1 - 0.434) / 2 = 0.657
method:   (1 - 0.270 + 1 - 0.364) / 2 = 0.683
```

The loss shifted errors between the two classes rather than removing them.
This is the signature of an **information-theoretic floor** at CIFAR-10
resolution: the dog/cat distinction is not linearly separable in the features
available at 32×32, so a local penalty can move the decision boundary but
cannot reduce the joint error. The class gap may even worsen because the
boundary moves away from the majority class.

## 5.3 Worst-class-weighted aggregation: the specified signal is weak

Step 3 weighted each client by `exp(-γ · recall_i(c_worst))` with γ=5.0,
where `c_worst` was the globally worst class. The global worst class was
airplane early and dog later. Because the formula gives the largest weight to
clients with **low** recall on the worst class, it upweights the clients least
able to represent that class. The gap fell only 0.443 → 0.431 (2.7%); dog
recall reached 0.292. Aggregation reweighting alone does not fix a class
representation problem.

## 5.4 Per-class distillation: the novel rung

Step 4 modified DEAL's KL term, weighting each class by `exp(r_i(c)/T)` with
T=1.0. Dog recall rose from 0.251 to 0.282 and frog from 0.694 to 0.706; the
gap fell 0.443 → 0.424 (4.3%). The best dog architecture was EfficientNet-B1
before and after, and its mean dog recall rose from 0.599 to 0.629, so the
best-dog architecture did not change. This is the strongest mechanism-specific
result in the ladder, but it is still far below the 30% closure threshold.

## 5.5 Federation amplifies, but does not create, the gap

The Standalone (local-only) run has a **38.6-point** gap, versus **44.3
points** for PHP-FL/linear. Federation therefore amplifies the class gap by
about 5.7 points, but most of the gap exists without any server aggregation.
The problem is not created by the FL protocol; it is inherited from the
data/model geometry and then modestly amplified.

## 5.6 Interpretation

The ladder does not support treating class_FM as a training objective. Four
different mechanisms—data rehearsal, a confusion-aware loss, aggregation
reweighting, and per-class distillation—each moved the gap by at most 7.9%
over one seed. The confusion-loss result is the clearest negative signal: it
reduced dog→cat errors while increasing the combined dog+cat error, which is
what a resolution-limited information floor looks like. Class_FM is a
**reporting metric**: it exposes a failure that client-FM hides, but the
failure is not reliably reducible by adding a scalar penalty to the local
objective. Reporting it is the contribution; optimizing it is not.
