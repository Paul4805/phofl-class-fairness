# RESULTS - scoped PHP-FL experiments

Scope: A fairness metrics; B aggregation reweighting; C correlated participation.  Smoke scale: 10 clients, 40 rounds, Fashion-MNIST, 1 seed.  E4-E8 were not run.  AM/FM use the repository's protocol: the best pre-training evaluation round over rounds 0..40.

Limitations: 10 clients x 1,000 train / 200 test samples, two official backbones (GoogLeNet, ResNet18), one seed.  This is a diagnostic smoke run, not a CIFAR-10 claim.  Wilcoxon has 10 pairs and minimum two-sided p=0.00195.

## A. Fairness is more than std

| exp             | method     | dataset      | pattern            | agg_mode   |   seed |   round |     AM |     FM |   min_acc |    p10 |   cvar10 |     CV |   jain |   rho_p_acc |
|:----------------|:-----------|:-------------|:-------------------|:-----------|-------:|--------:|-------:|-------:|----------:|-------:|---------:|-------:|-------:|------------:|
| B_agg           | PHPFL      | FashionMNIST | normal             | inv_N      |      0 |      40 | 0.5105 | 0.1615 |    0.2600 | 0.2915 |   0.2600 | 0.3164 | 0.9090 |      0.0667 |
| B_agg           | PHPFL      | FashionMNIST | normal             | inv_p      |      0 |      40 | 0.5160 | 0.1628 |    0.2600 | 0.2780 |   0.2600 | 0.3154 | 0.9095 |     -0.0182 |
| B_agg           | PHPFL      | FashionMNIST | normal             | plain      |      0 |      40 | 0.5175 | 0.1607 |    0.2400 | 0.2985 |   0.2400 | 0.3105 | 0.9120 |     -0.0061 |
| C_participation | PHPFL      | FashionMNIST | entropy_correlated | plain      |      0 |      40 | 0.5120 | 0.1805 |    0.1750 | 0.2425 |   0.1750 | 0.3526 | 0.8894 |      0.7212 |
| C_participation | PHPFL      | FashionMNIST | linear             | plain      |      0 |      40 | 0.5040 | 0.1439 |    0.2650 | 0.2875 |   0.2650 | 0.2856 | 0.9246 |     -0.7576 |
| C_participation | Standalone | FashionMNIST | linear             | none       |      0 |      40 | 0.4605 | 0.1856 |    0.1000 | 0.1990 |   0.1000 | 0.4030 | 0.8603 |     -0.7939 |

Lowest FM is PHPFL/linear/plain (FM=0.1439); best min-accuracy is PHPFL/linear/plain (0.2650); best CVaR10 is PHPFL/linear/plain (0.2650).  The FM winner also wins the tail metrics in this smoke run.  Spearman rho(p_i, accuracy) for the FM winner is -0.7576 (not positive).


## B. Aggregation reweighting

| exp   | agg_mode   |   client_id | arch      |    p_i |   N_i |    acc |   rare_acc |     AM |     FM |   seed |
|:------|:-----------|------------:|:----------|-------:|------:|-------:|-----------:|-------:|-------:|-------:|
| B     | plain      |           0 | googlenet | 0.3045 |    12 | 0.5100 |     0.0000 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           1 | resnet18  | 0.6957 |    26 | 0.3050 |     0.0000 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           2 | googlenet | 0.9482 |    38 | 0.7300 |     0.2000 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           3 | resnet18  | 0.8735 |    35 | 0.2400 |     0.0000 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           4 | googlenet | 0.5800 |    28 | 0.6350 |     0.0000 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           5 | resnet18  | 0.8528 |    33 | 0.6050 |     0.0541 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           6 | googlenet | 0.5821 |    27 | 0.4000 |     0.0000 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           7 | resnet18  | 0.4794 |    16 | 0.6750 |     0.0000 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           8 | googlenet | 0.4697 |    20 | 0.4150 |     0.1429 | 0.5175 | 0.1607 |      0 |
| B     | plain      |           9 | resnet18  | 0.6900 |    27 | 0.6600 |     0.0000 | 0.5175 | 0.1607 |      0 |
| B     | inv_p      |           0 | googlenet | 0.3045 |    12 | 0.4200 |     0.0000 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           1 | resnet18  | 0.6957 |    26 | 0.2800 |     0.0000 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           2 | googlenet | 0.9482 |    38 | 0.7500 |     0.2000 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           3 | resnet18  | 0.8735 |    35 | 0.2600 |     0.0000 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           4 | googlenet | 0.5800 |    28 | 0.6450 |     0.0000 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           5 | resnet18  | 0.8528 |    33 | 0.6250 |     0.0541 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           6 | googlenet | 0.5821 |    27 | 0.4150 |     0.0000 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           7 | resnet18  | 0.4794 |    16 | 0.6700 |     0.0000 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           8 | googlenet | 0.4697 |    20 | 0.4600 |     0.1429 | 0.5160 | 0.1628 |      0 |
| B     | inv_p      |           9 | resnet18  | 0.6900 |    27 | 0.6350 |     0.0000 | 0.5160 | 0.1628 |      0 |
| B     | inv_N      |           0 | googlenet | 0.3045 |    12 | 0.4750 |     0.0000 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           1 | resnet18  | 0.6957 |    26 | 0.2950 |     0.0000 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           2 | googlenet | 0.9482 |    38 | 0.7400 |     0.0000 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           3 | resnet18  | 0.8735 |    35 | 0.2600 |     0.0000 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           4 | googlenet | 0.5800 |    28 | 0.6100 |     0.0000 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           5 | resnet18  | 0.8528 |    33 | 0.6450 |     0.1351 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           6 | googlenet | 0.5821 |    27 | 0.3550 |     0.0000 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           7 | resnet18  | 0.4794 |    16 | 0.6750 |     0.0000 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           8 | googlenet | 0.4697 |    20 | 0.4300 |     0.1429 | 0.5105 | 0.1615 |      0 |
| B     | inv_N      |           9 | resnet18  | 0.6900 |    27 | 0.6200 |     0.0000 | 0.5105 | 0.1615 |      0 |

Per-client paired accuracy (rows = clients):

|   client_id |   inv_N |   inv_p |   plain |
|------------:|--------:|--------:|--------:|
|           0 |  0.4750 |  0.4200 |  0.5100 |
|           1 |  0.2950 |  0.2800 |  0.3050 |
|           2 |  0.7400 |  0.7500 |  0.7300 |
|           3 |  0.2600 |  0.2600 |  0.2400 |
|           4 |  0.6100 |  0.6450 |  0.6350 |
|           5 |  0.6450 |  0.6250 |  0.6050 |
|           6 |  0.3550 |  0.4150 |  0.4000 |
|           7 |  0.6750 |  0.6700 |  0.6750 |
|           8 |  0.4300 |  0.4600 |  0.4150 |
|           9 |  0.6200 |  0.6350 |  0.6600 |

- `plain`: mean=0.5175, std=0.1607, min=0.2400
- `inv_p`: mean=0.5160, std=0.1628, min=0.2600
- `inv_N`: mean=0.5105, std=0.1615, min=0.2600

## C. Correlated participation

| exp   | pattern            | method     |   client_id | arch      |    p_i |   N_i |    acc |   rare_acc |     AM |     FM |   seed |
|:------|:-------------------|:-----------|------------:|:----------|-------:|------:|-------:|-----------:|-------:|-------:|-------:|
| C     | linear             | PHPFL      |           0 | googlenet | 0.4556 |    19 | 0.5550 |     0.0000 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           1 | resnet18  | 0.6333 |    30 | 0.2900 |     0.0000 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           2 | googlenet | 0.2778 |     6 | 0.6500 |     0.0000 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           3 | resnet18  | 0.7222 |    28 | 0.2650 |     0.0000 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           4 | googlenet | 0.3667 |    16 | 0.6150 |     0.0000 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           5 | resnet18  | 0.5444 |    25 | 0.6250 |     0.0541 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           6 | googlenet | 0.9000 |    36 | 0.3550 |     0.0000 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           7 | resnet18  | 0.1000 |     7 | 0.6400 |     0.0000 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           8 | googlenet | 0.8111 |    33 | 0.4450 |     0.5714 | 0.5040 | 0.1439 |      0 |
| C     | linear             | PHPFL      |           9 | resnet18  | 0.1889 |     9 | 0.6000 |     0.0000 | 0.5040 | 0.1439 |      0 |
| C     | linear             | Standalone |           0 | googlenet | 0.4556 |    40 | 0.5700 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           1 | resnet18  | 0.6333 |    40 | 0.1000 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           2 | googlenet | 0.2778 |    40 | 0.7200 |     0.2000 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           3 | resnet18  | 0.7222 |    40 | 0.2100 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           4 | googlenet | 0.3667 |    40 | 0.5950 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           5 | resnet18  | 0.5444 |    40 | 0.5000 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           6 | googlenet | 0.9000 |    40 | 0.3350 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           7 | resnet18  | 0.1000 |    40 | 0.6100 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           8 | googlenet | 0.8111 |    40 | 0.4000 |     0.2857 | 0.4605 | 0.1856 |      0 |
| C     | linear             | Standalone |           9 | resnet18  | 0.1889 |    40 | 0.5650 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | PHPFL      |           0 | googlenet | 0.4916 |    21 | 0.5850 |     0.0000 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           1 | resnet18  | 0.4858 |    15 | 0.1750 |     0.0000 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           2 | googlenet | 0.7922 |    30 | 0.7300 |     0.0000 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           3 | resnet18  | 0.4095 |    18 | 0.2500 |     0.0000 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           4 | googlenet | 0.4787 |    24 | 0.6150 |     0.0000 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           5 | resnet18  | 0.5104 |    17 | 0.6200 |     0.0270 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           6 | googlenet | 0.4542 |    16 | 0.3800 |     0.0000 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           7 | resnet18  | 0.4964 |    16 | 0.6800 |     0.0000 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           8 | googlenet | 0.5992 |    27 | 0.4400 |     0.1429 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | PHPFL      |           9 | resnet18  | 0.7538 |    31 | 0.6450 |     0.0000 | 0.5120 | 0.1805 |      0 |
| C     | entropy_correlated | Standalone |           0 | googlenet | 0.4916 |    40 | 0.5700 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           1 | resnet18  | 0.4858 |    40 | 0.1000 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           2 | googlenet | 0.7922 |    40 | 0.7200 |     0.2000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           3 | resnet18  | 0.4095 |    40 | 0.2100 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           4 | googlenet | 0.4787 |    40 | 0.5950 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           5 | resnet18  | 0.5104 |    40 | 0.5000 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           6 | googlenet | 0.4542 |    40 | 0.3350 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           7 | resnet18  | 0.4964 |    40 | 0.6100 |     0.0000 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           8 | googlenet | 0.5992 |    40 | 0.4000 |     0.2857 | 0.4605 | 0.1856 |      0 |
| C     | entropy_correlated | Standalone |           9 | resnet18  | 0.7538 |    40 | 0.5650 |     0.0000 | 0.4605 | 0.1856 |      0 |

Per-client paired accuracy (rows = clients):

|   client_id |   ('entropy_correlated', 'PHPFL') |   ('entropy_correlated', 'Standalone') |   ('linear', 'PHPFL') |   ('linear', 'Standalone') |
|------------:|----------------------------------:|---------------------------------------:|----------------------:|---------------------------:|
|           0 |                            0.5850 |                                 0.5700 |                0.5550 |                     0.5700 |
|           1 |                            0.1750 |                                 0.1000 |                0.2900 |                     0.1000 |
|           2 |                            0.7300 |                                 0.7200 |                0.6500 |                     0.7200 |
|           3 |                            0.2500 |                                 0.2100 |                0.2650 |                     0.2100 |
|           4 |                            0.6150 |                                 0.5950 |                0.6150 |                     0.5950 |
|           5 |                            0.6200 |                                 0.5000 |                0.6250 |                     0.5000 |
|           6 |                            0.3800 |                                 0.3350 |                0.3550 |                     0.3350 |
|           7 |                            0.6800 |                                 0.6100 |                0.6400 |                     0.6100 |
|           8 |                            0.4400 |                                 0.4000 |                0.4450 |                     0.4000 |
|           9 |                            0.6450 |                                 0.5650 |                0.6000 |                     0.5650 |

Under `linear`, PHP-FL FM=0.1439, min=0.2650, rho(p,acc)=-0.758. Under `entropy_correlated`, FM=0.1805, min=0.1750, rho(p,acc)=+0.721. The entropy pattern raises mean accuracy slightly but shifts accuracy toward already-frequent clients: the tail collapses and the paired dispersion test is significant while the mean-accuracy test is not.  This is a fairness/tail effect, not a mean-accuracy collapse.

## Paired Wilcoxon tests

| metric             | comparison                               |   n |   statistic |   p_value |   median_diff | note                                     | significant_0.05   |
|:-------------------|:-----------------------------------------|----:|------------:|----------:|--------------:|:-----------------------------------------|:-------------------|
| accuracy           | plain vs inv_p [accuracy]                |  10 |     26.0000 |    0.9219 |       -0.0125 |                                          | False              |
| accuracy           | plain vs inv_N [accuracy]                |  10 |     16.0000 |    0.4688 |        0.0050 |                                          | False              |
| accuracy           | inv_p vs inv_N [accuracy]                |  10 |     17.0000 |    0.5703 |        0.0050 |                                          | False              |
| accuracy           | PHPFL vs Standalone [linear]             |  10 |      9.0000 |    0.0645 |        0.0325 |                                          | False              |
| accuracy           | PHPFL vs Standalone [entropy_correlated] |  10 |      0.0000 |    0.0020 |        0.0425 |                                          | True               |
| accuracy           | linear vs entropy [accuracy]             |  10 |     15.0000 |    0.4102 |       -0.0125 |                                          | False              |
| |acc - group mean| | linear vs entropy [dispersion]           |  10 |      7.5000 |    0.0410 |        0.0225 | larger = more unequal                    | True               |
| PHPFL - Standalone | linear vs entropy [advantage]            |  10 |     15.0000 |    0.4102 |        0.0125 | positive = advantage grows under entropy | False              |

Significant at 0.05: PHPFL vs Standalone [entropy_correlated] (p=0.001953); linear vs entropy [dispersion] (p=0.04102).

PHP-FL AM under linear: 0.5040; under entropy: 0.5120; paired median change: +0.0080.

## Artifacts

- `figures/A_fairness.png`
- `figures/B_aggregation_clients.png`
- `figures/B_aggregation_summary.png`
- `figures/C_participation_methods.png`
- `figures/C_participation_tail.png`
- `figures/C_participation_scatter.png`
