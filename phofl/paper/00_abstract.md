Federated learning fairness is usually summarized by client-FM, the standard
deviation of per-client accuracy. We show that this is a one-dimensional
projection of a two-dimensional fairness object. On five completed
model-heterogeneous CIFAR-10 runs, the PHP-FL/linear run has a **44.3-point**
best-to-worst class recall spread (frog 69.4%, dog 25.1%) while its client-FM is
0.0785; the single best-client-FM run has a 45.6-point spread. Across PHP-FL and
FedAPEN, class gaps exceed 40 points, placing the finding in the field-wide
category for the two MH-FL methods tested. Client-FM and class-level dispersion
rank the five runs differently, with FedAPEN 3rd by client-FM but 5th by class
gap; this is an observation, not a systematic correlation claim. We
additionally show that standard FL interventions (class-balanced rehearsal,
confusion-aware losses, worst-class-weighted aggregation, and per-class
distillation) reduce the gap by at most 7.9% over one seed, and that a
confusion-aware loss shifts errors between dog and cat without reducing their
combined rate, indicating an information-theoretic floor at CIFAR-10
resolution. We propose
**class-balanced FM** (`class_FM`), the standard deviation of per-class mean
recall, as a mandatory reporting metric alongside client-FM and worst-class
recall. `class_FM` is a reporting metric, not a training method. Support
filtering and AM-normalized results are reported.
