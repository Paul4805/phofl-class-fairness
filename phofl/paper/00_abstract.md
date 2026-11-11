Federated learning fairness is usually summarized by client-FM, the standard
deviation of per-client accuracy. We show that this is a one-dimensional
projection of a two-dimensional fairness object. Across an eight-cell CIFAR-10
comparison (PHP-FL, FedAPEN, FedFV, Standalone × two participation patterns ×
three seeds), best-to-worst class recall gaps are 0.35–0.60 while client-FM is
0.08–0.10: the class axis is as large as the accuracy axis it is summarized
away. Re-measuring with three seeds changes the published headline. PHP-FL/linear
has a class gap of **0.39 ± 0.08** (per-seed 0.32–0.47), so the earlier
44.3-point figure was a high seed draw; Standalone is statistically
indistinguishable at 0.389 ± 0.019; FedAPEN is consistently worst at
0.60 ± 0.09 in 6/6 per-seed comparisons. Client-FM and class gap still rank
runs oppositely: the worst-gap cell is also the best-client-FM cell in two of
three seeds. A conflict-aware aggregation rule (FedFV) has the best mean gap
and worst-class recall but only a directional 3.4-point margin over PHP-FL,
and we do not claim it as a fix. Standard interventions (class-balanced
rehearsal, confusion-aware losses, worst-class-weighted aggregation, per-class
distillation) reduce the gap by at most 7.9% over one seed, and a
confusion-aware loss shifts errors between dog and cat without reducing their
combined rate: full 10×10 matrices show the displaced mass reappearing as
dog/cat→bird confusion. A centralized 32×32 probe reaches half the federated
dog+cat joint error on the same pixels, so this is not an information floor at
CIFAR-10 resolution; the upsampled 64/96 probe degrades and cannot test one.
We propose
**class-balanced FM** (`class_FM`), the standard deviation of per-class mean
recall, as a mandatory reporting metric alongside client-FM and worst-class
recall. `class_FM` is a reporting metric, not a training method. Support
filtering and AM-normalized results are reported. Seed variance is reported for
every class-axis claim, and single-seed worst-class labels are shown to be
unstable.
