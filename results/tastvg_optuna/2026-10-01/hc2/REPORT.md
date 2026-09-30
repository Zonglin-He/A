# hc2: wide Optuna development search

32 source search / 16 disjoint-source confirmation, two orders, six conditions; all historical project exposure. No full-dataset parameter selection.

Selected trial 44: `{"lr": 0.10966593490131663, "rho": 0.19757784966059425, "teacher_temperature": 9.528864583977343}`.
Search corrupt nonexpert delta vIoU: default +1.230171 pp; selected +1.319067 pp. These are selection-biased development outcomes.

| Confirmation arm | Corrupt nonexpert delta vIoU pp [95% source CI] |
|---|---|
| default | -0.017878 [-0.030982, -0.004334] |
| selected | -0.014562 [-0.044606, +0.011469] |

All trials, failures, all-stream/clean/negative tails are retained. A small historically exposed confirmation panel does not establish a fresh full-test result. No automatic promotion or full-dataset rerun.
