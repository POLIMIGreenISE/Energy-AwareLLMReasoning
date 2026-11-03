# Bootstrap 95% CIs for headline numbers

Percentile bootstrap over queries, B=2000, N=1710 (easy 1327 / hard 383), seed 12345.

Router/oracle CIs are **conditional on the fitted classifier**: P(think) is the full-data 5-fold out-of-fold prediction (identical to notebook 03), held fixed while queries are resampled — evaluation uncertainty, not retraining variance. The measured J/token load curve is likewise held fixed for RQ4.

| headline | point | 2.5% | 97.5% |
|---|---|---|---|
| no_think mean energy (mWh/q) | 341 | 325 | 355 |
| think mean energy (mWh/q) | 2776 | 2608 | 2954 |
| think / no_think energy ratio | 8.2 | 7.7 | 8.6 |
| easy (overthinking): no_think acc | 0.964 | 0.954 | 0.972 |
| easy (overthinking): think acc | 0.970 | 0.962 | 0.978 |
| easy (overthinking): thinking gain (pp) | +0.6 | -0.1 | +1.4 |
| hard (underthinking): no_think acc | 0.385 | 0.343 | 0.428 |
| hard (underthinking): think acc | 0.782 | 0.742 | 0.821 |
| hard (underthinking): thinking gain (pp) | +39.7 | +35.0 | +44.4 |
| router EAA-AUC | 0.899 | 0.887 | 0.910 |
| oracle frontier EAA-AUC | 0.929 | 0.918 | 0.938 |
| % of oracle gain captured | 68.5 | 64.7 | 72.3 |
| acc gain of tau(load) (pp) | +8.4 | +7.1 | +9.8 |
| budget-100mWh acc gap (pp) | +9.4 | +8.1 | +10.9 |
