# O2: Regime-Coherent Online Transfer

Positive mean transfer appears in this fixed coherent-regime screen. Its size, changed cases and conditional intervals delimit the support; locality is not uniquely identified as the cause.

Same original16 C3 sources in five persistent regimes: frame drop, frame freeze, motion blur, occlusion and exposure, each using the already-fixed5% random-burst coverage. Same source-only O1 hash order restricted to these16, repeated across conditions. Four expert positions1/5/9/13 per stream; state resets to zero for each condition.

## Primary: non-expert arrivals

| Condition | N | Frozen/Budgeted tIoU (%) | Online tIoU (%) | Delta t (pp) | Frozen/Budgeted vIoU (%) | Online vIoU (%) | Delta v (pp) | Changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frame_drop_5 | 12 | 30.9312 | 31.0042 | 0.0730 | 17.9215 | 17.9429 | 0.0214 | 3 |
| frame_freeze_5 | 12 | 28.7007 | 28.7007 | 0.0000 | 17.5552 | 17.5552 | 0.0000 | 0 |
| motion_blur_5 | 12 | 29.5281 | 29.5281 | 0.0000 | 17.7451 | 17.7451 | 0.0000 | 0 |
| occlusion_5 | 12 | 26.4714 | 27.0452 | 0.5737 | 15.8555 | 16.0611 | 0.2056 | 1 |
| exposure_5 | 12 | 28.5402 | 28.5402 | 0.0000 | 16.5749 | 16.5749 | 0.0000 | 0 |
| macro | 60 | 28.8343 | 28.9637 | 0.1293 | 17.1304 | 17.1758 | 0.0454 | 4 |

Macro equally weights the five conditions. Since their source order and expert schedule coincide, uncertainty resamples12 non-expert parents AFTER averaging each parent across five conditions. This is60 cells, not60 independent sources. All-arrival secondary macro uses16 parents. The10000-sample seed20260929 paired intervals condition on these fixed shared trajectories; they do not rerun state evolution.

| Condition | Delta t 95% interval (pp) | Delta v 95% interval (pp) | Full agreement Online / Native | Final weight norm | Expert losses down |
|---|---:|---:|---:|---:|---:|
| frame_drop_5 | [+0.0000, +0.2190] | [+0.0000, +0.0642] | 0/12 / 0/12 | 0.01136213 | 4/4 |
| frame_freeze_5 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1/12 / 1/12 | 0.00189893 | 4/4 |
| motion_blur_5 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0/12 / 0/12 | 0.00147897 | 4/4 |
| occlusion_5 | [+0.0000, +1.7211] | [+0.0000, +0.6168] | 0/12 / 0/12 | 0.00748164 | 4/4 |
| exposure_5 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0/12 / 0/12 | 0.00139716 | 4/4 |
| macro | [+0.0000, +0.3734] | [+0.0000, +0.1319] | 1/60 / 1/60 | per-stream | 20/20 |

## Four-arm reference and negative tails

| Arm | Logical expert calls | Nonexpert macro tIoU (%) | Nonexpert macro vIoU (%) | All-arrival macro tIoU (%) | All-arrival macro vIoU (%) |
|---|---:|---:|---:|---:|---:|
| Frozen | 0 | 28.8343 | 17.1304 | 35.0408 | 15.8454 |
| Budgeted Rerank | 20 | 28.8343 | 17.1304 | 37.0651 | 16.8205 |
| Online Slow-Fast | 20 | 28.9637 | 17.1758 | 37.1621 | 16.8546 |
| Full Rerank | 80 | 34.3094 | 20.0385 | 41.1714 | 19.0016 |

Online versus Budgeted nonexpert vIoU gains/losses/unchanged: 2/0/58. >5pp cell harms: 0. All per-condition and parent-macro harms remain in SUMMARY.json.

Full Rerank is a higher-budget reference, not a correctness oracle. Its agreement is diagnostic only. The public ROWS.json retains all candidate metrics, teacher/native/arrival scores, selections, and both useful and harmful cases.

| Changed nonexpert | Parent | Condition | Frozen t/v (%) | Online t/v (%) | Full t/v (%) |
|---|---|---|---:|---:|---:|
| 2 | Q13 | frame_drop_5 | 0.0000/0.0000 | 0.0000/0.0000 | 0.0000/0.0000 |
| 3 | Q09 | frame_drop_5 | 15.7664/5.1749 | 16.6423/5.4316 | 14.0146/4.6161 |
| 15 | Q03 | frame_drop_5 | 0.0000/0.0000 | 0.0000/0.0000 | 0.0000/0.0000 |
| 8 | Q16 | occlusion_5 | 28.8770/10.6912 | 35.7616/13.1584 | 31.9527/11.7972 |

## Mechanism and limits

The method is unchanged from O1: raw768D final temporal hidden start/end/span mean, native-envelope base score, zero w+b, FP64 CPU strict-pair logistic loss, one plain SGD .001 per expert, alpha1. Emit expert output before writing future state; no current teacher read at nonexpert arrivals. No normalization, gate, prototype, LR/alpha search, or H/model updates. All20 sparse teacher outputs were loaded separately; extra60 reference outputs were accessed only after all online states/output were sealed.

READOUT_DIAGNOSTIC.json reports the ACTUAL O2 residual advantage and native gap toward Full on each nonexpert arrival. It does not amplify residuals or rerun a trajectory. O1.1 showed that old O1 directions could move choices when amplified; that does not guarantee sufficient readout scale after four unchanged O2 writes. Zero gain with unchanged selections is therefore not a decisive representation-failure result.

O2 changes stream construction while keeping method parameters fixed, as requested, but its16-source/four-write design differs from O1's32-source/eight-write design. Outcomes cannot uniquely attribute any difference to regime coherence. These historically exposed development sources and single fixed orders do not establish robust deployment transfer.

## Verification, cost and reproduction

80 captures exactly match old pixels/preprocessed inputs, boxes and both offset logits; frozen model state is unchanged. Native zero-state selection holds on80/80. Independent NumPy audit reconstructs all80 arrival states and20 SGD writes, including all five resets, feature definitions, hash chronology and absence of nonexpert teacher reads. Predictions for all arms were sealed before reading original16 GT keys. Each candidate metric was checked using two independent metric routines, and all80 critic scores were recomputed from proposals. Public audit reconstructs486 mean/interval checks and matched controls.

New GPU process time: 92.961s (80 frozen inputs /160 offset forwards, including loading/CPU decoding). No new expert computation;20 online +60 reference evidence items reused exactly. Online CPU loop: 0.099s. Cache reuse preserves logical expert budgets and does not represent fresh-input end-to-end latency. Initial shell wrapper permission failure occurred before Python/model startup and was resolved by invoking the existing wrapper with bash; retained launch_failure.log. No changes to system permissions.

Existing O1 three CPU contract tests pass. Resource metadata includes exact stage allocation and any model-worker failure. Production CURRENT and old evidence are unchanged. No follow-on experiment launched.

Reproduction: run_tastvg_regime_online_capture_v1.py prepare/run -> run_tastvg_regime_online_teacher_v1.py online -> run_tastvg_regime_online_v1.py online -> audit_tastvg_regime_online_v1.py -> teacher full -> online full -> analyze_tastvg_regime_online_v1.py -> report_tastvg_regime_online_v1.py. Public readback: audit_tastvg_regime_online_public_v1.py <result-directory>. Raw media, labels, features and state tensors stay local.
