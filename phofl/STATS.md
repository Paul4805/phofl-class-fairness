# STATS - paired Wilcoxon signed-rank tests

Unit of analysis: client.  Each pair is matched by `client_id`; seeds are not treated as replicates (one seed in this smoke run). Two-sided test, zero differences use Wilcoxon's default handling.

| metric             | comparison                               |   n |   statistic |   p_value |   median_diff | note                                     | significant_0.05   |
|:-------------------|:-----------------------------------------|----:|------------:|----------:|--------------:|:-----------------------------------------|:-------------------|
| accuracy           | plain vs inv_p [accuracy]                |  10 |    26.00000 |   0.92188 |      -0.01250 |                                          | False              |
| accuracy           | plain vs inv_N [accuracy]                |  10 |    16.00000 |   0.46875 |       0.00500 |                                          | False              |
| accuracy           | inv_p vs inv_N [accuracy]                |  10 |    17.00000 |   0.57031 |       0.00500 |                                          | False              |
| accuracy           | PHPFL vs Standalone [linear]             |  10 |     9.00000 |   0.06445 |       0.03250 |                                          | False              |
| accuracy           | PHPFL vs Standalone [entropy_correlated] |  10 |     0.00000 |   0.00195 |       0.04250 |                                          | True               |
| accuracy           | linear vs entropy [accuracy]             |  10 |    15.00000 |   0.41016 |      -0.01250 |                                          | False              |
| |acc - group mean| | linear vs entropy [dispersion]           |  10 |     7.50000 |   0.04102 |       0.02250 | larger = more unequal                    | True               |
| PHPFL - Standalone | linear vs entropy [advantage]            |  10 |    15.00000 |   0.41016 |       0.01250 | positive = advantage grows under entropy | False              |

Spearman rho(p_i, per-client accuracy) is -0.758 under linear and +0.721 under entropy_correlated.  C result: mean accuracy moves 0.5040 -> 0.5120 (paired p=0.4102), while FM moves 0.1439 -> 0.1805 and min accuracy moves 0.2650 -> 0.1750; the paired per-client dispersion test is p=0.04102.  The effect is a tail/fairness degradation, not a significant mean collapse.
With 10 clients the smallest attainable two-sided p-value is 0.00195.  A non-significant mean-accuracy result is not evidence of no effect.  The dispersion row is the fairness-specific paired test: it asks whether the per-client accuracy distribution became more unequal, which the mean-accuracy test cannot detect.  A is descriptive (one baseline vector) and therefore has no paired test.  No multiple-comparison correction is applied; these are exploratory smoke tests and the dispersion p=0.041 is marginal.
