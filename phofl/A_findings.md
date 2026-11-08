# A - fairness metric recomputation

| exp             | method     | dataset      | pattern            | agg_mode   |   seed |   round |     AM |     FM |   min_acc |    p10 |   cvar10 |     CV |   jain |   rho_p_acc |
|:----------------|:-----------|:-------------|:-------------------|:-----------|-------:|--------:|-------:|-------:|----------:|-------:|---------:|-------:|-------:|------------:|
| B_agg           | PHPFL      | FashionMNIST | normal             | inv_N      |      0 |      40 | 0.5105 | 0.1615 |    0.2600 | 0.2915 |   0.2600 | 0.3164 | 0.9090 |      0.0667 |
| B_agg           | PHPFL      | FashionMNIST | normal             | inv_p      |      0 |      40 | 0.5160 | 0.1628 |    0.2600 | 0.2780 |   0.2600 | 0.3154 | 0.9095 |     -0.0182 |
| B_agg           | PHPFL      | FashionMNIST | normal             | plain      |      0 |      40 | 0.5175 | 0.1607 |    0.2400 | 0.2985 |   0.2400 | 0.3105 | 0.9120 |     -0.0061 |
| C_participation | PHPFL      | FashionMNIST | entropy_correlated | plain      |      0 |      40 | 0.5120 | 0.1805 |    0.1750 | 0.2425 |   0.1750 | 0.3526 | 0.8894 |      0.7212 |
| C_participation | PHPFL      | FashionMNIST | linear             | plain      |      0 |      40 | 0.5040 | 0.1439 |    0.2650 | 0.2875 |   0.2650 | 0.2856 | 0.9246 |     -0.7576 |
| C_participation | Standalone | FashionMNIST | linear             | none       |      0 |      40 | 0.4605 | 0.1856 |    0.1000 | 0.1990 |   0.1000 | 0.4030 | 0.8603 |     -0.7939 |

Lowest FM is PHPFL/linear/plain (FM=0.1439); best min-accuracy is PHPFL/linear/plain (0.2650); best CVaR10 is PHPFL/linear/plain (0.2650).  The FM winner also wins the tail metrics in this smoke run.  Spearman rho(p_i, accuracy) for the FM winner is -0.7576 (not positive).
