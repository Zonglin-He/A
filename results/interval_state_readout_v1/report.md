# 整段状态可分性

Status: Historical negative; not adopted. All outputs completed and independently scored; no method adopted.

All128 queries have repeated development exposure. HC32/Vid32 are parent-isolated historical development checks, not untouched confirmation. Every complete panel sealed before this panel's new GT scoring. GT-derived error/rescue subsets are offline diagnosis only.

## Signed effects relative to Global6 + original BoxFixed

| Cohort | Method | ΔtIoU pp | Δfull vIoU pp | Δfixed-track vIoU pp | tIoU .5 rescue/loss | Changed intervals |
|---|---|---:|---:|---:|---:|---:|
| HC64 | StateHidden | -11.5712 | -6.2497 | -6.2934 | 1/10 | 52 |
| HC64 | StateSpatial | -8.7930 | -4.3775 | -4.5765 | 5/14 | 62 |
| HC64 | StateJoint | -5.3105 | -2.7464 | -2.8852 | 1/5 | 39 |
| HC64 | StatePermuted | -7.5698 | -4.0197 | -4.1145 | 0/7 | 40 |
| HC64 | StateGlobal | -10.2509 | -5.4993 | -5.5498 | 1/7 | 47 |
| HC64 | PriorMatched | -0.8274 | -0.4517 | -0.5232 | 0/2 | 39 |
| HC64 | RandomMatched | -5.3614 | -3.3790 | -3.4003 | 2/5 | 39 |
| HC64 | OldJoint10 | -0.1175 | -0.0263 | -0.0175 | 0/0 | 5 |
| HC32 | StateHidden | -3.4201 | -1.1013 | -1.2076 | 0/1 | 27 |
| HC32 | StateSpatial | -4.5241 | -0.0053 | -0.3143 | 1/5 | 32 |
| HC32 | StateJoint | +1.0132 | +0.7581 | +0.4923 | 1/0 | 15 |
| HC32 | StatePermuted | +0.7966 | +1.0317 | +0.7659 | 1/0 | 16 |
| HC32 | StateGlobal | -2.1115 | -0.1720 | -0.2857 | 0/1 | 24 |
| HC32 | PriorMatched | -0.0717 | -0.0412 | -0.0356 | 0/0 | 15 |
| HC32 | RandomMatched | -0.7609 | -0.3819 | -0.3436 | 0/2 | 15 |
| HC32 | OldJoint10 | +0.4082 | +0.2664 | +0.2732 | 0/0 | 2 |
| Vid32 | StateHidden | -8.2106 | -5.2432 | -5.2197 | 0/4 | 27 |
| Vid32 | StateSpatial | -2.9415 | -1.7330 | -1.7192 | 3/1 | 29 |
| Vid32 | StateJoint | -7.0374 | -4.6945 | -4.6790 | 0/3 | 16 |
| Vid32 | StatePermuted | -6.8845 | -4.9682 | -4.9585 | 1/3 | 16 |
| Vid32 | StateGlobal | -5.7579 | -3.4631 | -3.4397 | 1/2 | 23 |
| Vid32 | PriorMatched | -0.2304 | +0.0019 | +0.0053 | 0/0 | 16 |
| Vid32 | RandomMatched | -0.9911 | -1.0457 | -1.0138 | 2/1 | 16 |
| Vid32 | OldJoint10 | +0.0732 | +0.0028 | +0.0028 | 0/0 | 1 |

All per-query signs, boundary errors, severe losses, observed/unobserved spatial scores, confidence intervals and Source-relative outcomes remain in report.json and signed_queries.tsv. Candidate oracle is diagnostic, never an inference input.

## Interpretation

Case audit: among eight HC64 wrong-but-rescuable queries, the correct candidate topped the previous evidence score only once. Five had favorable evidence relative to the incumbent but insufficient to offset native expected-IoU loss. Seven incumbent starts were early; only one to four observations fell within the true event. Whole-interval detrended prefix/event/suffix state separation was then tested with true/permuted/global spatial controls. It often selected a visually distinct but semantically wrong event. State separability is not the queried event likelihood.

Do not treat this failed proxy as proof that generic features cannot support temporal correction.

## Evidence and scope

Official/dense comparisons: 2560; max absolute discrepancy 1.89e-15. Training/GPU/new experts0. Original inputs, predictions, receipts and seals are preserved privately; no raw boxes/features/weights/query or GT coordinates are published.

The running full evaluation is a separate immutable protocol. All434 frozen sources and its plan remain unchanged. Current full-domain results remain local until both domains complete.
