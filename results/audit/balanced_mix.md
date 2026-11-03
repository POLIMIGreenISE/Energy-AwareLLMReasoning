# Workload-mix sensitivity — balanced 50/50 easy/hard

N=1710 (1327 easy weighted down to match 383 hard).

| policy | EAA (balanced) | % oracle gain |
|--|--|--|
| A text router | 0.8061 | 71% |
| P p_need rule | 0.7330 | 32% |
| oracle frontier | 0.8591 | 100% |

Never 0.674 @ 445 mWh; Always 0.876 @ 4754 mWh.
tau(load) lambda=1/Wh: static 0.674 -> adaptive 0.861 at batch 64 (+18.7pp).
