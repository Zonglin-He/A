# 内部辅助头无校准排序

Status: Retired at explicit user instruction; never a generic method. All outputs completed and independently scored; no method adopted.

All128 queries have repeated development exposure. HC32/Vid32 are parent-isolated historical development checks, not untouched confirmation. Every complete panel sealed before this panel's new GT scoring. GT-derived error/rescue subsets are offline diagnosis only.

## Signed effects relative to Global6 + original BoxFixed

| Cohort | Method | ΔtIoU pp | Δfull vIoU pp | Δfixed-track vIoU pp | tIoU .5 rescue/loss | Changed intervals |
|---|---|---:|---:|---:|---:|---:|
| HC64 | ActionOnly | -6.4831 | -3.2991 | -3.3493 | 0/5 | 46 |
| HC64 | ActionJoint | -6.3125 | -2.9871 | -3.0859 | 1/6 | 48 |
| HC64 | ActionPermuted | -2.3454 | -1.2326 | -1.2487 | 1/1 | 19 |
| HC64 | ActionRaw | -6.4831 | -3.2991 | -3.3493 | 0/5 | 46 |
| HC64 | JointRaw | -6.3125 | -2.9871 | -3.0859 | 1/6 | 48 |
| HC64 | UnconstrainedAction | -6.8881 | -3.5505 | -3.5976 | 0/6 | 47 |
| HC64 | UnconstrainedJoint | -6.2524 | -2.9835 | -3.0940 | 1/6 | 50 |
| HC64 | PriorMatched | -1.8521 | -1.1720 | -1.2361 | 0/4 | 48 |
| HC64 | RandomMatched | -6.7892 | -4.1293 | -4.2289 | 2/9 | 48 |
| HC64 | OldJoint10 | -0.1175 | -0.0263 | -0.0175 | 0/0 | 5 |
| HC64 | StateJoint | -5.3105 | -2.7464 | -2.8852 | 1/5 | 39 |
| HC32 | ActionOnly | -3.2647 | -1.3106 | -1.3041 | 0/2 | 24 |
| HC32 | ActionJoint | -3.8264 | -1.4874 | -1.4874 | 0/3 | 29 |
| HC32 | ActionPermuted | -1.1709 | -0.2083 | -0.1786 | 0/0 | 7 |
| HC32 | ActionRaw | -3.2647 | -1.3106 | -1.3041 | 0/2 | 24 |
| HC32 | JointRaw | -3.8264 | -1.4874 | -1.4874 | 0/3 | 29 |
| HC32 | UnconstrainedAction | -3.7521 | -1.7113 | -1.7047 | 0/2 | 25 |
| HC32 | UnconstrainedJoint | -4.1043 | -1.7109 | -1.7109 | 0/3 | 29 |
| HC32 | PriorMatched | -1.3044 | -0.5876 | -0.6048 | 0/1 | 29 |
| HC32 | RandomMatched | -2.0623 | -0.5424 | -0.5051 | 0/4 | 29 |
| HC32 | OldJoint10 | +0.4082 | +0.2664 | +0.2732 | 0/0 | 2 |
| HC32 | StateJoint | +1.0132 | +0.7581 | +0.4923 | 1/0 | 15 |
| Vid32 | ActionOnly | -4.9267 | -3.2573 | -3.2673 | 0/1 | 26 |
| Vid32 | ActionJoint | -3.0446 | -2.4950 | -2.4853 | 0/1 | 28 |
| Vid32 | ActionPermuted | -1.3468 | -1.1361 | -1.1361 | 0/1 | 1 |
| Vid32 | ActionRaw | -4.9267 | -3.2573 | -3.2673 | 0/1 | 26 |
| Vid32 | JointRaw | -3.0446 | -2.4950 | -2.4853 | 0/1 | 28 |
| Vid32 | UnconstrainedAction | -5.9624 | -3.7574 | -3.7633 | 0/2 | 26 |
| Vid32 | UnconstrainedJoint | -4.0803 | -2.9950 | -2.9813 | 0/2 | 28 |
| Vid32 | PriorMatched | -0.0662 | +0.1352 | +0.1386 | 0/0 | 28 |
| Vid32 | RandomMatched | -0.6234 | -0.7747 | -0.7427 | 2/1 | 28 |
| Vid32 | OldJoint10 | +0.0732 | +0.0028 | +0.0028 | 0/0 | 1 |
| Vid32 | StateJoint | -7.0374 | -4.6945 | -4.6790 | 0/3 | 16 |

All per-query signs, boundary errors, severe losses, observed/unobserved spatial scores, confidence intervals and Source-relative outcomes remain in report.json and signed_queries.tsv. Candidate oracle is diagnostic, never an inference input.

## Interpretation

This completed historical branch uses a model-specific internal actionness head or its outputs and is **excluded from the requested generic algorithm**. It is retained solely for truthful provenance and negative-result reproducibility. No further runs, reuse of its scores for gates, or production integration are authorized.

Likelihood calibration and interval rank were separately tested. Both failed development gates; higher frame AUC did not translate into correct interval selection. Absolute weighted-BCE calibration encouraged short intervals. Rank removed the absolute offset issue but did not establish task-relevant ranking. These findings do not rehabilitate this architecture-specific direction.

## Evidence and scope

Official/dense comparisons: 3328; max absolute discrepancy 1.89e-15. Training/GPU/new experts0. Original inputs, predictions, receipts and seals are preserved privately; no raw boxes/features/weights/query or GT coordinates are published.

The running full evaluation is a separate immutable protocol. All434 frozen sources and its plan remain unchanged. Current full-domain results remain local until both domains complete.
