# vidstg: wide Optuna development search

32 source search / 16 disjoint-source confirmation, two orders, six conditions; all historical project exposure. No full-dataset parameter selection.

Selected trial 5: `{"lr": 0.005, "rho": 0.05, "teacher_temperature": 0.05}`.
Search corrupt nonexpert delta vIoU: default +0.017387 pp; selected +0.786177 pp. These are selection-biased development outcomes.

| Confirmation arm | Corrupt nonexpert delta vIoU pp [95% source CI] |
|---|---|
| default | +0.017697 [+0.004217, +0.032117] |
| selected | +0.166143 [-0.011859, +0.365932] |

All trials, failures, all-stream/clean/negative tails are retained. A small historically exposed confirmation panel does not establish a fresh full-test result. No automatic promotion or full-dataset rerun.
