# 内部辅助头区间似然

Status: Retired at explicit user instruction; never a generic method. All outputs completed and independently scored; no method adopted.

All128 queries have repeated development exposure. HC32/Vid32 are parent-isolated historical development checks, not untouched confirmation. Every complete panel sealed before this panel's new GT scoring. GT-derived error/rescue subsets are offline diagnosis only.

## Signed effects relative to Global6 + original BoxFixed

| Cohort | Method | ΔtIoU pp | Δfull vIoU pp | Δfixed-track vIoU pp | tIoU .5 rescue/loss | Changed intervals |
|---|---|---:|---:|---:|---:|---:|
| HC64 | ActionOnly | -15.1670 | -7.1777 | -7.1691 | 1/18 | 64 |
| HC64 | ActionJoint | -14.9477 | -7.1109 | -7.1077 | 1/17 | 64 |
| HC64 | ActionPermuted | -4.7082 | -1.8407 | -1.7960 | 2/6 | 33 |
| HC64 | ActionRaw | -6.3999 | -3.3564 | -3.4685 | 2/8 | 44 |
| HC64 | JointRaw | -6.6274 | -3.2455 | -3.3933 | 3/9 | 43 |
| HC64 | UnconstrainedAction | -18.7818 | -9.4960 | -9.4338 | 1/21 | 64 |
| HC64 | UnconstrainedJoint | -18.0745 | -9.1266 | -9.0608 | 1/19 | 64 |
| HC64 | PriorMatched | -2.6975 | -1.4402 | -1.5235 | 0/5 | 64 |
| HC64 | RandomMatched | -8.9483 | -5.1616 | -5.3039 | 2/10 | 64 |
| HC64 | OldJoint10 | -0.1175 | -0.0263 | -0.0175 | 0/0 | 5 |
| HC64 | StateJoint | -5.3105 | -2.7464 | -2.8852 | 1/5 | 39 |
| HC32 | ActionOnly | -7.7966 | -2.9254 | -2.9222 | 0/6 | 31 |
| HC32 | ActionJoint | -8.2240 | -3.0594 | -3.0417 | 0/6 | 31 |
| HC32 | ActionPermuted | -4.0424 | -0.3739 | -0.3984 | 0/3 | 21 |
| HC32 | ActionRaw | -2.4823 | -0.5957 | -0.6577 | 0/2 | 17 |
| HC32 | JointRaw | -2.4997 | -1.1412 | -1.2033 | 0/2 | 15 |
| HC32 | UnconstrainedAction | -7.8620 | -2.5310 | -2.6578 | 0/6 | 32 |
| HC32 | UnconstrainedJoint | -7.4635 | -2.0312 | -2.3183 | 1/6 | 32 |
| HC32 | PriorMatched | -1.7410 | -0.6694 | -0.6877 | 0/2 | 31 |
| HC32 | RandomMatched | -2.5874 | -0.4946 | -0.4583 | 0/5 | 31 |
| HC32 | OldJoint10 | +0.4082 | +0.2664 | +0.2732 | 0/0 | 2 |
| HC32 | StateJoint | +1.0132 | +0.7581 | +0.4923 | 1/0 | 15 |
| Vid32 | ActionOnly | -4.2727 | -3.3874 | -3.3892 | 0/1 | 29 |
| Vid32 | ActionJoint | -2.1321 | -1.4343 | -1.4325 | 0/0 | 29 |
| Vid32 | ActionPermuted | -1.4907 | -1.4049 | -1.4049 | 0/1 | 3 |
| Vid32 | ActionRaw | -2.5461 | -2.0861 | -2.0884 | 0/1 | 7 |
| Vid32 | JointRaw | -1.6702 | -1.3684 | -1.3805 | 0/0 | 8 |
| Vid32 | UnconstrainedAction | -6.0625 | -4.2058 | -4.2037 | 0/2 | 29 |
| Vid32 | UnconstrainedJoint | -3.9219 | -2.2528 | -2.2469 | 0/1 | 29 |
| Vid32 | PriorMatched | +0.5041 | +0.1057 | +0.1047 | 0/0 | 29 |
| Vid32 | RandomMatched | -0.4560 | -0.7412 | -0.7327 | 2/1 | 29 |
| Vid32 | OldJoint10 | +0.0732 | +0.0028 | +0.0028 | 0/0 | 1 |
| Vid32 | StateJoint | -7.0374 | -4.6945 | -4.6790 | 0/3 | 16 |

All per-query signs, boundary errors, severe losses, observed/unobserved spatial scores, confidence intervals and Source-relative outcomes remain in report.json and signed_queries.tsv. Candidate oracle is diagnostic, never an inference input.

## Interpretation

This completed historical branch uses a model-specific internal actionness head or its outputs and is **excluded from the requested generic algorithm**. It is retained solely for truthful provenance and negative-result reproducibility. No further runs, reuse of its scores for gates, or production integration are authorized.

Likelihood calibration and interval rank were separately tested. Both failed development gates; higher frame AUC did not translate into correct interval selection. Absolute weighted-BCE calibration encouraged short intervals. Rank removed the absolute offset issue but did not establish task-relevant ranking. These findings do not rehabilitate this architecture-specific direction.

## Evidence and scope

Official/dense comparisons: 3328; max absolute discrepancy 1.89e-15. Training/GPU/new experts0. Original inputs, predictions, receipts and seals are preserved privately; no raw boxes/features/weights/query or GT coordinates are published.

The running full evaluation is a separate immutable protocol. All434 frozen sources and its plan remain unchanged. Current full-domain results remain local until both domains complete.
