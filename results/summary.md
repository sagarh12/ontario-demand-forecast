| Model | MAE (MW) | RMSE (MW) | MAPE (%) | Skill vs best baseline |
|---|---:|---:|---:|---:|
| HistGradientBoosting (selected on validation) | 502 | 641 | 2.94 | +33.8% |
| Random forest | 530 | 684 | 3.08 | +30.0% |
| Ridge regression | 536 | 673 | 3.23 | +29.3% |
| Baseline: same hour yesterday | 758 | 1030 | 4.58 | +0.0% |
| Baseline: hybrid | 926 | 1375 | 5.51 | -22.1% |
| Baseline: same hour last week | 1197 | 1673 | 7.05 | -57.9% |

Test year 2025, 8759 hours. Best baseline on the test year: Baseline: same hour yesterday. Skill = 1 - MAE_model / MAE_best_baseline.

**Validation vs test.** Models were selected on 2024 and scored once on 2025.

| Model | Validation MAE (2024) | Test MAE (2025) |
|---|---:|---:|
| Baseline: same hour yesterday | 751 | 758 |
| Baseline: same hour last week | 1229 | 1197 |
| Baseline: hybrid | 849 | 926 |
| Ridge regression | 522 | 536 |
| Random forest | 466 | 530 |
| HistGradientBoosting | 435 | 502 |

**By segment** (test MAE, MW):

| | HistGradientBoosting | Random forest | Ridge regression | Baseline: same hour yesterday |
|---|---:|---:|---:|---:|
| Mean bias, whole year (pred - actual) | -391 | -396 | -351 | -6 |
| MAE, weekday | 459 | 508 | 498 | 743 |
| MAE, weekend | 609 | 587 | 631 | 795 |
| MAE, non-holiday | 500 | 531 | 533 | 757 |
| MAE, holiday | 562 | 505 | 643 | 790 |

By hour, HistGradientBoosting is best at hour 4 (402 MW) and worst at hour 12 (609 MW). By month it is best in month 5 (293 MW) and worst in month 7 (754 MW).

**Hardest days.** The 10 hottest test days by daily max temperature (mean 27.6 °C) and the 10 coldest by daily min temperature (mean -12.3 °C):

| | HistGradientBoosting | Random forest | Ridge regression | Baseline: same hour yesterday |
|---|---:|---:|---:|---:|
| Heat-wave days: MAE | 1086 | 1213 | 776 | 1434 |
| Heat-wave days: bias (pred - actual) | -1022 | -1168 | -666 | -1338 |
| Heat-wave days: highest prediction (actual peak 24862) | 23438 | 23301 | 25202 | 24789 |
| Cold-snap days: MAE | 577 | 709 | 467 | 691 |
| Cold-snap days: bias (pred - actual) | -533 | -677 | -431 | -470 |
| Cold-snap days: highest prediction (actual peak 21940) | 21158 | 20755 | 21308 | 21701 |

Worst five days for HistGradientBoosting:

| Date | MAE | Bias | Max temp (°C) |
|---|---:|---:|---:|
| 2025-06-22 | 1922 | -1922 | 31.5 |
| 2025-08-10 | 1542 | -1527 | 32.2 |
| 2025-12-26 | 1433 | -1433 | -5.4 |
| 2025-03-29 | 1403 | -1403 | 6.4 |
| 2025-08-16 | 1364 | -1364 | 30.5 |

**Permutation importance** (HistGradientBoosting, test year, top 10):

| Feature | MAE increase when shuffled (MW) |
|---|---:|
| `lag_24` | 837 ± 7 |
| `temp` | 539 ± 3 |
| `day_of_week` | 164 ± 3 |
| `hour_sin` | 136 ± 3 |
| `hour_cos` | 124 ± 2 |
| `hdd` | 123 ± 1 |
| `humidity` | 48 ± 2 |
| `lag_168` | 38 ± 2 |
| `prev_day_std` | 38 ± 1 |
| `doy_cos` | 25 ± 1 |
