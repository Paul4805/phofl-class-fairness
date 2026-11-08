# 2. Related work

## 2.1 Client-level fairness in federated learning

Most FL fairness work defines fairness across clients. q-FedAvg [Li et al.,
ICLR 2020] reweights client updates to improve the worst-client objective;
AFL [Mohri et al., ICML 2019] minimizes an agnostic distribution over clients;
FedFV [Wang et al., IJCAI 2021] adds a fairness-aware aggregation rule;
FedHEAL adds a heterogeneity-aware fairness objective. These methods are
evaluated primarily with client-accuracy dispersion, minimum client accuracy,
or a fairness index over clients. The evaluation object is typically the vector
of client accuracies, with classes marginalized out.

Model-heterogeneous FL adds a second source of inequality: clients run
different architectures, so their local models and feature spaces differ.
PHP-FL addresses participation-probability inconsistency in this setting with
a dual-end alignment loss and an individual-specific parameter update. FedAPEN
[Qin et al., KDD 2023] uses a shared auxiliary model and a learned ensemble
weight for personalized cross-silo FL. Both methods are evaluated on
client-level average accuracy and client dispersion. Neither reports a
client-by-class recall matrix.

## 2.2 Class-level fairness and long-tailed recognition

Class imbalance and long-tailed recognition have a mature centralized
literature. Class-balanced loss [Cui et al., CVPR 2019], LDAM [Cao et al.,
NeurIPS 2019], and group-distributionally robust optimization [Sagawa et al.,
ICLR 2020] all make the class or group axis explicit. Group-fairness work in
supervised learning similarly reports per-group error rates and worst-group
accuracy rather than only an average. The common lesson is that a single
aggregate can hide a large group-level spread.

FL inherits both axes at once. A client's test set may contain only a subset of
classes; a class may be rare in the global population but frequent for a few
clients. Averaging client accuracies first, as FM does, weights clients equally
and classes implicitly by their within-client frequency. The resulting number
can be stable while class recall varies by tens of percentage points.

## 2.3 The gap

Existing MH-FL evaluations report client fairness, and existing class-fairness
evaluations are mostly centralized. The two axes are rarely reported together.
This paper does not introduce a new training algorithm; it makes the two-axis
object explicit for MH-FL and proposes a support-aware reporting practice. The
empirical question is whether client-FM and class-level dispersion agree on
the same runs. On our five completed runs they still order the systems differently, but five
runs are not enough to establish a systematic relationship.
