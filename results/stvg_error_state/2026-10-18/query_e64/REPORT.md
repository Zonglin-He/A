# Frozen-decoder query-only E adaptation64

Exact historical HC32/Vid32, frozen Source reset and QR256=0 for every arm/query. E and original four planned positions are fixed. Only external query residual differentiates through frozen decoder; no pretrained or LN parameter changes. Final outputs are the Student, with no E overlay. This is spatial adaptation, not temporal or cross-video learning.

## Results

| Arm | HC vIoU % | HC unobserved actual sIoU % | Vid vIoU % | Vid unobserved actual sIoU % |
|---|---:|---:|---:|---:|
| Source_MBR | 28.004527 | 49.048024 | 16.616909 | 33.657953 |
| E_MBR | 27.558379 | 48.345122 | 22.453299 | 43.213676 |
| Sparse_OPD | 25.155821 | 45.852909 | 16.945818 | 33.921315 |
| Dense_OPD | 26.973980 | 47.954783 | 16.845936 | 33.784055 |
| Fixed_rollout | 28.037336 | 49.340311 | 16.678095 | 33.671281 |
| Dense_Direct | 29.299386 | 51.225227 | 17.128366 | 35.753347 |

| Dense OPD minus | HC delta v pp | HC95% CI | Vid delta v pp | Vid95% CI |
|---|---:|---|---:|---|
| Source_MBR | -1.030547 | [-3.587876,+0.609603] | +0.229027 | [+0.058042,+0.428392] |
| E_MBR | -0.584399 | [-3.765736,+2.961110] | -5.607363 | [-9.581533,-2.256494] |
| Sparse_OPD | +1.818159 | [-0.217919,+5.612482] | -0.099882 | [-0.276869,+0.020243] |
| Fixed_rollout | -1.063356 | [-3.608192,+0.574462] | +0.167842 | [+0.004643,+0.356707] |
| Dense_Direct | -2.325406 | [-5.893288,+0.273120] | -0.282430 | [-0.760192,-0.010234] |

## Scope and interpretation

Each comparison changes the declared support, resampling or objective only. Sparse and Dense use the SAME E teacher; fixed rollout uses initial actions with the original likelihood evaluated at current policy. Direct uses the same E and mean5L1+2GIoU. All use identical domain-specific inherited learning rate and predetermined rounds, but actual parameter/output movements differ and are reported.

Dense-vs-Sparse tests coverage; Dense-vs-Fixed tests resampling; Dense-vs-Direct tests OPD objective; Dense-vs-E tests whether adaptation earns current-query deployment benefit. Source benefit alone does not establish superiority over directly applying E. See DECISION.json for per-comparator two-domain endpoint checks. No automatic tuning, expansion or promotion follows this development panel.

This cohort has historical exposure and source calibration overlap already disclosed in prior reports. Parent bootstrap intervals describe this fixed episodic panel; they do not establish independent confirmation. E pseudoboxes contain no new unobserved-frame expert information. Geometry IoU is not event-boundary feedback. Negative tails, no-update and unknown-support samples remain in denominators.

## Cost and verification

GPU lease 620.208652s; 64 exact frozen Source prefix captures; 5865 query-only Adam updates; zero new DINO calls, zero pretrained/LN updates and zero source training. Existing E/DINO acquisition cost is reused, not erased. Pilot timing is included in the original64, not extra samples.

All64 six-output predictions sealed before new target GT scoring. Original Source/E metrics exact; all time readouts/tIoU identical. Independent audit checks QR zero/reset, unchanged LN, original action/reward/gradient/Adam math, actual resampling vs frozen reuse, full two-pass readout, metrics and parent intervals. Fixed first-parent planned4 plus uniformly chosen unobserved4 actual-pixel panels per domain are generated for root inspection; no new semantic labels claimed. Previous LN experiment remains paused; second backbone not started.

[All signed results](RESULTS.json) · [Actual protocol](ACTUAL_PROTOCOL.json) · [Costs](ACTUAL_COSTS.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Decision](DECISION.json) · [Contrasts](QUERY_ONLY_CONTRASTS.svg) · [All parents](QUERY_ONLY_ALL_PARENTS.svg).

## Post-hoc sensitivity and audit precision

The primary means and all parents above remain unchanged. Leave-one-out sensitivity diagnoses concentration and is not a selection or exclusion rule.
- hc2, Dense minus Source_MBR: largest positive H119 +8.371813pp; largest negative H128 -34.280431pp; leave-one-parent mean range [-1.333849,+0.042030]pp.
- hc2, Dense minus Sparse_OPD: largest positive H008 +58.389029pp; largest negative H115 -1.994816pp; leave-one-parent mean range [-0.006708,+1.941158]pp.
- vidstg, Dense minus Source_MBR: largest positive V016 +1.869882pp; largest negative V038 -0.840642pp; leave-one-parent mean range [+0.176096,+0.263533]pp.
- vidstg, Dense minus Sparse_OPD: largest positive V082 +0.457465pp; largest negative V038 -2.325538pp; leave-one-parent mean range [-0.117861,-0.028087]pp.

The audit originally compared saved FP32 Adam moments with independently accumulated FP64 moments using an absolute-only tolerance. Its original failure is preserved. The corrected audit records absolute/relative errors and uses rtol2e-6 plus atol2e-5 for moments; parameter-step checks remain unchanged (maximum error below2e-8). No scientific inputs, update paths, predictions or scores changed or reran.

Root actually inspected both aggregate plots and16 fixed planned/unobserved input-output frames. This is an engineering pixel check, not independent semantic identity labeling. Private media are excluded from publication.
