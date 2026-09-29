# S1 — Persistent spatial Reverse-KL OPD

Completed / independently audited. Parameters really updated, persisted, and changed future predictions; practical task change in this first fixed setting is negligible. Keep the positive S0.6 critic result separate from the online adaptation result.

| Group / subset | Cells / sources | Online−Frozen sIoU pp | Source95% CI pp | Online−Frozen vIoU pp | Source95% CI pp |
|---|---|---:|---|---:|---|
| corruption / nonexpert | 60 / 12 | 0.000228 | [0.000091, 0.000389] | 0.000089 | [-0.000026, 0.000214] |
| corruption / all | 80 / 16 | 0.000195 | [0.000090, 0.000318] | 0.000069 | [-0.000017, 0.000164] |
| corruption / expert | 20 / 4 | 0.000096 | [0.000000, 0.000191] | 0.000009 | [-0.000001, 0.000028] |
| clean / nonexpert | 12 / 12 | 0.000057 | [-0.000121, 0.000229] | -0.000004 | [-0.000138, 0.000107] |
| clean / all | 16 / 16 | -0.000035 | [-0.000291, 0.000171] | 0.000000 | [-0.000102, 0.000084] |
| clean / expert | 4 / 4 | -0.000310 | [-0.001148, 0.000217] | 0.000012 | [0.000000, 0.000034] |

## Executed online contract

Six independent16-arrival streams (clean plus five original transient5% corruptions), ascending original roster, source reset only at stream boundaries. Same Vid-source TA checkpoint,1792D spatial query/final-LN parameters. Primary output is central policy BEFORE current expert/update. Scheduled expert positions0/4/8/12, others expert-free. Source0 expert masks empty, so6 no-op scheduled arrivals and18 effective updates. First three nonexpert arrivals per stream precede the first effective update; DIAGNOSTICS also reports the9-source later-nonexpert subset.

Every scheduled arrival regenerates9 detached student candidates around CURRENT state, using unchanged fixed antithetic directions and absolute radius1.57658019. No source-checkpoint candidate replay for changed central policies. Read current cached expert mask only after candidates exist, derive detached scalar IoU rewards. Student-to-student native-coefficient L1+GIoU compatibility, tauE=tauS=1, Reverse-KL, one plain SGD step eta.005. This rate reused the numerical existing Vid-source private-interface value; optimizer/loss differ, so it is a first setting rather than an established optimum. No new expert, teacher-coordinate regression, H update, margin gate, optimizer momentum, clipping, replay, line search or best-state selection.

The teacher shapes finite-support compatibility probabilities; this distribution is a surrogate, not a native normalized tube likelihood. Frozen and BudgetedRerank baselines are retained; BudgetedRerank only selects source candidates at scheduled expert arrivals and equals Frozen on the primary nonexpert subset. S1 never substitutes an expert-selected tube for its central prediction. Online-vs-budgeted/all-arrival metrics are in SUMMARY.

All18 updates reduce their same-arrival fixed-support KL. Query gradient nonzero 18/18 and LN gradient nonzero 18/18. Gradient norm median 0.01336365; actual SGD norm median 6.683082e-05, min 1.0147896e-05, max 0.0011599112; median step/probe-radius ratio 4.2389737e-05. q probability-range median 0.0025322251; p range median 0.020441491. These are numerical diagnostics, not a causal isolation of rate, temperature, geometry or reward strength.

24 candidate sets regenerated, 12 centered on outputs differing from source,18 SGD steps inherited by later samples. Four full native reinsertion checks with learnt states pass.96 state-chain entries and6 resets checked; final source checkpoint restored. Frozen H remains valid because encoder parameters never update.

## Interpretation and limits

Corruption future-nonexpert sIoU change is about+.000228pp and vIoU+.000089pp. This is numerically measurable but negligible practical utility; do not market a tiny positive s CI as an effective method. Clean changes are likewise tiny. The probe-scale critic signal did not turn into a useful parameter displacement/task gain under one tau1/SGD.005 step and this sparse18-update schedule. This does not prove critic failure, OPD impossibility, or adequate hyperparameter optimization. Do not silently increase learning rate/steps/temperature or add a gate after reading scores.

Same16 repeatedly exposed sources, primary12 nonexpert source positions; six streams share source content and each trajectory is sequentially dependent. Source bootstrap is descriptive and does not establish independent online-stream generalization. No temporal/joint run or production promotion.

GPU process 47.92s,18 backward calls,216 regenerated candidate tubes,0 new backbone captures or experts.3 new S1 CPU contracts,96 state links,18 independent KL/SGD reconstructions,192 dual metrics,4 learnt-state reinsertions and public scalar/CI audit. An audit-only float64-vs-FP32 KL tolerance initially failed (maximum2.76e-7 vs2e-7); changed numerical audit tolerance to1e-6 and recorded the measured error. No objective/gradient/prediction changed and no GPU rerun. See AUDIT_NUMERICAL_RECOVERY.json.
