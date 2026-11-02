# Kaizen Optimize: results on real data (2026-11-02 22:34)

## 1. Cutting data (model calibrated on real C45 turning experiments)

Roughness model: R² 0.95 on 40 measurements, leave-one-run-out error 29 % (2-level design, corners are extrapolated in this check). Tool life: T = 565 · vc^-0.32 min (VB = 0.3 mm, f 0.1, ap 0.2), valid 70–240 m/min.

| Part | Strategy | Median gap to model optimum | 90th percentile gap | Meets spec |
|---|---|---|---|---|
| A shaft | DOE + response surface | 16.0 % | 63.7 % | 100 % |
| A shaft | Random (Latin hypercube) | 166.1 % | 251.4 % | 100 % |
| A shaft | Bayesian optimization (GP) | 9.7 % | 11.9 % | 100 % |
| B flange | DOE + response surface | 13.7 % | 53.5 % | 100 % |
| B flange | Random (Latin hypercube) | 186.9 % | 292.3 % | 100 % |
| B flange | Bayesian optimization (GP) | 8.7 % | 11.6 % | 100 % |
| C pin | DOE + response surface | 5.5 % | 11.0 % | 100 % |
| C pin | Random (Latin hypercube) | 109.6 % | 177.0 % | 87 % |
| C pin | Bayesian optimization (GP) | 4.2 % | 7.5 % | 93 % |
| D axle | DOE + response surface | 18.7 % | 74.5 % | 100 % |
| D axle | Random (Latin hypercube) | 194.3 % | 294.0 % | 100 % |
| D axle | Bayesian optimization (GP) | 11.2 % | 19.6 % | 100 % |

| Part | Handbook €/part | Recommended €/part | Model optimum | Recommended setting |
|---|---|---|---|---|
| A shaft | 9.36 | 5.94 | 5.49 | 240 m/min, 0.137 mm, 2 pass(es), Ra 2.66 µm |
| B flange | 7.77 | 4.27 | 3.94 | 237 m/min, 0.135 mm, 2 pass(es), Ra 2.64 µm |
| C pin | 3.71 | 2.28 | 2.23 | 240 m/min, 0.115 mm, 1 pass(es), Ra 2.17 µm |
| D axle | 18.28 | 10.97 | 10.02 | 240 m/min, 0.137 mm, 2 pass(es), Ra 2.66 µm |

## 2. Scheduling (benchmark: 10 instances, 54 jobs, 5 machines)

Model check: 9 of 9 small instances solved to proven optimality give exactly the published best value.

| Method | Mean gap to best known | Median gap |
|---|---|---|
| HGA (published) | 0.5 % | 0.0 % |
| Iterated greedy (ours) | 2.6 % | 2.2 % |
| CP (published) | 4.4 % | 4.0 % |
| SA (published) | 16.7 % | 16.7 % |
| MDD (published) | 49.9 % | 49.3 % |
| Gemini (published) | 63.7 % | 64.9 % |
| ATCS (published) | 65.1 % | 64.3 % |
| ATCS rule (ours) | 66.2 % | 59.4 % |
| GPT4.1 (published) | 72.9 % | 59.4 % |
| MILP (published) | 84.3 % | 37.8 % |
| EDD rule (ours) | 150.8 % | 138.6 % |
| EDD (published) | 210.3 % | 212.0 % |

This week's order book (medium-1), cycle times x 0.60 with the new cutting data:

| Cycle times | Method | Weighted tardiness | Late jobs | Setup | Makespan |
|---|---|---|---|---|---|
| handbook | EDD | 111127 | 54 | 1732 | 892 |
| handbook | ATCS | 67631 | 51 | 815 | 1032 |
| handbook | Iterated greedy | 50222 | 51 | 803 | 648 |
| optimized | EDD | 75849 | 54 | 1667 | 637 |
| optimized | ATCS | 103638 | 50 | 873 | 1313 |
| optimized | Iterated greedy | 29455 | 47 | 802 | 478 |

## 3. Maintenance (SCANIA Component X, real fleet data)

23550 vehicles, 2272 repairs, 21278 censored. Weibull β 2.41 (95 % CI 2.34–2.49), η 692. Best replacement age 425 (95 % CI 408–441); hold-out Kaplan-Meier optimum 471.

| Policy | Cost per 1000 time units | On hold-out vehicles | Ends in breakdown |
|---|---|---|---|
| Run to failure | 8316 € | – | 100 % |
| Replace at characteristic life (692) | 7035 € | – | 63 % |
| Optimized age (425) | 6302 € | 5759 € | 27 % |

## 4. Spare-part inventory (321 real parts, tested on 12 unseen months)

| Policy | Cost per month | Fill rate | Average stock |
|---|---|---|---|
| Rule of thumb | 2771 € | 98.4 % | 1382 units |
| Textbook (normal + EOQ) | 3816 € | 100.0 % | 2680 units |
| Optimized on full history | 3163 € | 99.0 % | 2114 units |
| Optimized on last 12 months | 2739 € | 96.3 % | 1420 units |
