# 7. Conclusion

Client-FM is not a sufficient fairness report for model-heterogeneous
federated learning. Across eight half-scale cells and three seeds, the PHP-FL
family contains the best client-FM. Its linear run has a **0.39 ± 0.08**
class-recall spread: the single-seed 44.3-point figure was a high draw, and the
spread is not distinguishable from Standalone's 0.389 ± 0.019. FedAPEN shows a
**0.60 ± 0.09** gap and is worse than PHP-FL in 6/6 per-seed comparisons, which
promotes the observation to field-wide across the two MH-FL methods tested. The
class axis is so large under every participation pattern tested, but the
pattern effect is method-dependent rather than invariant: it moves the gap by
up to 15 points and reverses sign between FedAPEN and FedFV. The runs also rank
differently on client-FM and class gap; FedAPEN is 3rd by client-FM but 5th by
class gap, and this inversion repeats in every seed, with the worst-gap cell
also the best-client-FM cell in two of three. We therefore recommend reporting

```
class_FM = std_c( mean_i recall_{i,c} )
```

alongside client-FM and worst-class recall. `class_FM` is a reporting metric,
not an objective; it should not replace client-FM. The original participation
hypothesis is closed and is not revived. Limitations are three seeds, half-scale
training (~8x less than the paper), four CNNs, 10 clients, and support filtering
at >=10 samples per client-class cell. Standalone is tied with PHP-FL on the
class gap, so the earlier raw-gap comparison is dropped rather than normalized.
A four-rung diagnostic
ladder—class-balanced rehearsal, confusion-aware loss, worst-class-weighted
aggregation, and per-class distillation—reduced the gap by at most 7.9% over
one seed, so `class_FM` should be reported but not optimized as an objective.
Future work should add a second seed and a full-scale audit before any
universal claim.
