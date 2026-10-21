# HC64 state-OPD comparison

The original 128-query plan was reduced by the user to the first 64 parent videos in the fixed development manifest, one query per parent. The original method configuration and per-query budgets remain unchanged. All baseline means below are recomputed on these same 64 queries; no old 128-query aggregate is used as a comparison.

Exposed development panel: VidSTG to HC2, 64 parent videos, one query each. All methods use the exact saved Strong interval. All queries are retained; stopping follows the predeclared reward-based acceptance rules and evaluation allowance, without GT selection.

SourceStrong uses the saved frozen Source spatial trajectory with Strong temporal inference; it is not the original Source temporal readout. ERefStrong uses the saved single-ReferDINO initialization with the same Strong interval.

| Method | Full vIoU (%) | tIoU (%) | Observed-node sIoU (%) | Unobserved-node sIoU (%) |
|---|---:|---:|---:|---:|
| SourceStrong | 29.1530 | 58.6043 | 49.4198 | 47.6990 |
| ERefStrong | 31.0918 | 58.6043 | 52.2680 | 50.7055 |
| GeometryOPD | 31.3868 | 58.6043 | 53.1654 | 51.2469 |
| StateOPD | 31.0942 | 58.6043 | 52.2826 | 50.7248 |
| Direct | 31.0965 | 58.6043 | 52.2786 | 50.7883 |
| DirectMC | 31.0938 | 58.6043 | 52.2811 | 50.7800 |
| DirectFullRef | 31.1023 | 58.6043 | 52.4444 | 50.7469 |

Observed and unobserved sIoU use actual sampled nodes with GT; all eight expert positions are excluded from unobserved sIoU. These metrics are independent of temporal truncation. Interpolation is used only by the existing complete-video metric, not counted as new observations.

| Comparison | vIoU delta pp [95% CI] | Improved / declined / unchanged | vIoU drops >5 pp | Unobserved sIoU delta pp [95% CI] | Unobserved drops >5 pp |
|---|---:|---:|---:|---:|---:|
| StateOPD-ERefStrong | +0.0025 [-0.0004, 0.0055] | 41 / 22 / 1 | 0 | +0.0193 [0.0040, 0.0394] | 0 |
| StateOPD-GeometryOPD | -0.2925 [-0.6695, 0.0017] | 17 / 46 / 1 | 1 | -0.5221 [-1.0640, -0.0691] | 3 |
| Direct-ERefStrong | +0.0047 [-0.0131, 0.0200] | 37 / 25 / 2 | 0 | +0.0828 [-0.0058, 0.2160] | 0 |
| Direct-GeometryOPD | -0.2903 [-0.6644, 0.0030] | 16 / 47 / 1 | 1 | -0.4586 [-0.9818, -0.0306] | 2 |
| DirectMC-ERefStrong | +0.0020 [-0.0025, 0.0067] | 38 / 24 / 2 | 0 | +0.0745 [0.0005, 0.2092] | 0 |
| DirectMC-GeometryOPD | -0.2930 [-0.6695, 0.0008] | 17 / 46 / 1 | 1 | -0.4669 [-0.9857, -0.0408] | 2 |
| DirectFullRef-ERefStrong | +0.0105 [0.0010, 0.0214] | 41 / 22 / 1 | 0 | +0.0414 [0.0119, 0.0774] | 0 |
| DirectFullRef-GeometryOPD | -0.2845 [-0.6607, 0.0092] | 17 / 46 / 1 | 1 | -0.5000 [-1.0450, -0.0401] | 3 |
| StateOPD-Direct | -0.0022 [-0.0159, 0.0146] | 28 / 35 / 1 | 0 | -0.0635 [-0.1779, 0.0132] | 0 |
| StateOPD-DirectMC | +0.0004 [-0.0023, 0.0033] | 28 / 35 / 1 | 0 | -0.0552 [-0.1704, 0.0056] | 0 |
| StateOPD-DirectFullRef | -0.0080 [-0.0171, -0.0007] | 24 / 39 / 1 | 0 | -0.0221 [-0.0520, 0.0048] | 0 |
| Direct-DirectMC | +0.0027 [-0.0121, 0.0145] | 32 / 30 / 2 | 0 | +0.0083 [-0.0300, 0.0420] | 0 |
| Direct-DirectFullRef | -0.0058 [-0.0185, 0.0015] | 26 / 37 / 1 | 0 | +0.0414 [-0.0332, 0.1621] | 0 |
| DirectMC-DirectFullRef | -0.0085 [-0.0167, -0.0022] | 25 / 38 / 1 | 0 | +0.0331 [-0.0466, 0.1725] | 0 |

Intervals use 10,000 paired parent-video bootstrap draws. All positive, negative, zero-overlap, and unchanged queries remain in the anonymous result JSON. Valid-query counts are reported per metric; queries without GT at an expert observation are retained in complete vIoU.

The four new methods share the locked 8 x 4 temporary state space. GeometryOPD uses the older 8 x 256 state and is a saved method reference, not a matched optimizer control. This experiment does not establish superiority over every Direct update in the older full state space. Direct and DirectFullRef allow up to 660 point attempts; StateOPD and DirectMC allow up to 40 rounds. All share an allowance of at most 1,321 full-trajectory evaluations, including initialization and every backtracking validation. They need not consume the same allowance. The fixed backtracking scales are 1, 0.5, 0.25, 0.125, and 0.0625; the first utility increase above 1e-8 is accepted, otherwise the state and Adam buffers are restored. Point methods stop after all scales fail; OPD/MC may use the next prespecified sampling seed until their round or evaluation allowance is exhausted. Backward counts and actual wall times differ.

## Declining queries

The JSON retains every query. This table shows the union of the ten largest vIoU declines for each new method relative to ERef+Strong; values are percentage points.

| Query | StateOPD | Direct | DirectMC | DirectFullRef |
|---|---:|---:|---:|---:|
| H007 | -0.0177 | -0.0399 | -0.0235 | -0.0392 |
| H010 | -0.0036 | -0.0085 | -0.0046 | -0.0106 |
| H013 | -0.0025 | -0.0091 | -0.0045 | -0.0090 |
| H014 | -0.0146 | -0.0209 | -0.0271 | -0.0193 |
| H024 | -0.0194 | -0.0472 | -0.0281 | -0.0476 |
| H036 | -0.0417 | -0.1010 | -0.0665 | -0.0977 |
| H040 | +0.0020 | -0.0094 | -0.0078 | +0.0083 |
| H041 | -0.0046 | -0.0152 | -0.0041 | -0.0190 |
| H045 | -0.0053 | +0.0013 | -0.0098 | +0.0025 |
| H051 | -0.0131 | -0.0175 | -0.0178 | -0.0146 |
| H052 | -0.0042 | -0.3832 | -0.0506 | -0.0477 |
| H056 | -0.0043 | -0.0100 | -0.0062 | -0.0105 |

## Recorded costs

Observed incremental trial costs with existing Source/teacher caches; not standalone end-to-end inference latency

| Method | Seconds/query | Attempts | Accepted updates | Full trajectory evaluations | Native offset decoder calls | Decoder backwards | Reward evaluations | Mean per-query peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StateOPD | 18.0335 | 39.9531 | 39.7188 | 1320.3438 | 2640.6875 | 0.0000 | 1320.3438 | 156.10 |
| Direct | 16.8578 | 420.0312 | 419.3906 | 844.4375 | 1688.8750 | 420.0312 | 844.4375 | 260.55 |
| DirectMC | 32.5961 | 39.8281 | 38.6562 | 1319.9844 | 2639.9688 | 1274.5000 | 1319.9844 | 260.64 |
| DirectFullRef | 22.1041 | 457.6406 | 457.0312 | 919.2500 | 1838.5000 | 457.6406 | 919.2500 | 261.80 |

The geometric baseline is reused from exact sealed outputs; its old runtime is not presented as a new timing. Equal evaluation allowances do not imply equal consumed work, wall time or equal backward computation. Configuration and code are bound by the recorded hashes. The original formal VidSTG task and prior reward results are separate and are not modified.

## Outcome

This fixed mask-projection and bounded-state construction did not produce useful new complete-localization gains. StateOPD changes complete vIoU by only +0.0025 pp relative to ERef+Strong, and by -0.2925 pp relative to the saved geometric OPD. Within the matched 32-dimensional state space, its differences from Direct and DirectMC are -0.0022 and +0.0004 pp, respectively; neither establishes an OPD advantage. DirectFullRef is higher than StateOPD by 0.0080 pp, itself far below a practically useful gain.

No new arm has a complete-vIoU drop above 5 pp relative to ERef+Strong. That protection comes with almost no average task gain and cannot by itself establish successful correction. Relative to geometric OPD, StateOPD loses more than 5 pp on one query and more than 5 pp of unobserved-node sIoU on three queries; those lost corrections are retained in the result JSON.

The registered geometric method remains unchanged. The four new arms jointly change the reward, state parameterization and protection compared with the older geometric method, so their gap cannot be assigned to the reward alone. The same-state controls directly test the OPD update mechanism within this fixed new construction. This exposed HC64 screen does not determine the best configuration of any method family. No follow-up sweep or formal-task resume was performed.

## Saved-state interpretation

The post-run CPU audit reads only saved trajectories, utilities and the existing score table. All 256 arm-query trace checks passed; protected utility and guard recomputation errors were at most 1.11e-16. No decoder replay, new candidate, or new GT scoring was used for this interpretation. Reproduce the audit with `python -B scripts/state_opd/audit.py` after sealed predictions and scoring; aggregated measurements and anonymous cases are in [dynamics.json](dynamics.json).

| Measurement | StateOPD | Direct | DirectMC | DirectFullRef |
|---|---:|---:|---:|---:|
| Mean same-node IoU of final boxes versus ERef (no GT) | 0.99484 | 0.98874 | 0.99254 | 0.98945 |
| Mean absolute corner displacement (% normalized image coordinate) | 0.0585 | 0.1171 | 0.0781 | 0.1115 |
| Mean content L2 | 1.354 | 3.352 | 2.119 | 3.466 |
| Mean protected-utility increase | 0.002974 | 0.004497 | 0.003173 | 0.004635 |
| Queries with an active guard on the saved path | 4/64 | 9/64 | 5/64 | 8/64 |

All 64 queries increase utility in each arm, so a missing update is not the explanation. Most outputs remain close to ERef, but a few local nodes move substantially; the averages must not be described as every frame being unchanged. Guard inactivity on most paths does not support blaming trajectory protection for the entire null result.

StateOPD has ESS >=31 out of 32 in 2,486/2,557 sampled rounds, with median ESS 31.99995. Candidate utility standard deviation divided by the fixed temperature has median 0.001235. Candidate weights are therefore nearly uniform in most rounds. This is a measured limitation of the present candidate/utility/temperature combination, not proof that temperature alone is the cause: Adam rescales gradients, and the basis, radius, decoder response and round counts were not independently varied.

The lost geometric corrections are substantial on individual examples:

| Anonymous query | GeometryOPD minus ERef vIoU pp | StateOPD minus ERef vIoU pp | StateOPD minus GeometryOPD vIoU pp |
|---|---:|---:|---:|
| H014 | +8.9822 | -0.0146 | -8.9968 |
| H003 | +3.1972 | +0.0132 | -3.1840 |
| H031 | +2.2223 | +0.0047 | -2.2176 |

These are missed corrections relative to the older method, not equally large new losses relative to ERef. All 64 signed rows and all pairwise results remain available in [hc64.json](hc64.json).
