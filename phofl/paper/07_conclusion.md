# 7. Conclusion

Client-FM is not a sufficient fairness report for model-heterogeneous
federated learning. Across five half-scale runs, the PHP-FL family contains
the best client-FM. Its linear run still has a **44.3-point** class-recall
spread, and the single best-FM run is worse at 45.6 points. FedAPEN shows a
**52.9-point** gap, which promotes the observation to field-wide across the two
MH-FL methods tested. The class axis
is structural: changing the participation pattern moves class-std by only
0.0015 for PHP-FL and 0.0084 for Standalone. The five runs also rank differently
on client-FM and class gap; FedAPEN is 3rd by client-FM but 5th by class gap,
and this inversion survives AM normalization. We therefore recommend reporting

```
class_FM = std_c( mean_i recall_{i,c} )
```

alongside client-FM and worst-class recall. `class_FM` is a reporting metric,
not an objective; it should not replace client-FM. The original participation
hypothesis is closed and is not revived. Limitations are one seed, half-scale
training (~8x less than the paper), four CNNs, 10 clients, and support filtering
at >=10 samples per client-class cell. AM normalization removes the raw
Standalone gap advantage, so that claim is not made. A four-rung diagnostic
ladder—class-balanced rehearsal, confusion-aware loss, worst-class-weighted
aggregation, and per-class distillation—reduced the gap by at most 7.9% over
one seed, so `class_FM` should be reported but not optimized as an objective.
Future work should add a second seed and a full-scale audit before any
universal claim.
