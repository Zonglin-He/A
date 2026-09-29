# DESTA Dev16 sIoU decline: saved-evidence diagnosis

2026-09-29. Status: readback completed; DESTA remains paused by user. No GPU,
model forward/backward, optimizer, native prediction, new GT scoring, media
access, or new annotation-pool access was performed. Another experiment may
continue independently. This review does not authorize any restart or repair run.

The subject is the completed 27-configuration Dev16 experiment, not the incomplete
Dev64 expansion. Frozen PTD4B/B1 and Train-only QC R16 were used on 16 previously
exposed source-validation queries from 16 parents. The historical vIoU-best arm
is K1/radius .03/temporal weight 2/spatial weight 1. It was selected using exposed
Dev16 GT after prediction seals; it is not a fresh validation success.

## What actually declined

| Metric (%) | B1 | Historical vIoU-best arm | Difference (pp) |
|---|---:|---:|---:|
| tIoU | 39.517189 | 40.605776 | +1.088586 |
| sIoU | 36.243945 | 30.200662 | -6.043283 |
| vIoU | 23.041295 | 17.856283 | -5.185012 |

sIoU has 4 positive / 7 negative / 5 unchanged queries, median change 0. The
largest single decline contributes -5.498766 pp to the 16-parent mean: 90.9897%
of its net -6.043283 pp. This is a concentration description, not permission to
remove the sample. The other 15 have mean -0.580818 pp; all 16 remain in results.
The existing uncorrected descriptive sIoU CI is [-18.923831,+2.842859] pp.

## The main extreme is lost box support after a temporal shift

Case Q11 changes sIoU 87.980262% -> 0. Its already scored GT physical interval is
[38,106); exactly one observed frame, 73, lies in that interval, and the saved
report records exactly one valid GT spatial-support frame. B1 predicts sampled
positions [2,11], physical [73,404), including a valid box at frame 73. The final
updated prediction instead uses positions [9,12], physical [330,441), and no
longer produces a box at frame 73. Its semantic reference remains unchanged;
format and all output box geometries remain valid.

This precisely explains the zero spatial score without attributing an unseen
coordinate error to the omitted frame. In this implementation sIoU averages IoU
on a **fixed GT event-and-box-valid frame set**; a missing predicted box is zero.
The denominator does not switch to the newly predicted interval. The predicted
interval determines where native boxes exist, so temporal mistakes can reduce
sIoU even when boxes emitted elsewhere remain geometrically valid.

The saved UniversalVTG pseudo-interval is [201.248810,1102.497681], disjoint from
that already scored GT interval. Thus this query has concrete incorrect temporal
evidence, not merely low expert confidence. Both gradients entered a shared
correction; without the unrun T-only/S-only counterfactual, we cannot quantify
how much of the interval jump was caused by each gradient separately.

## Real coordinate degradation also exists

| Case | sIoU before -> after (%) | Saved endpoint evidence |
|---|---:|---|
| Q01 | 20.982105 -> 2.539335 | All 3 GT-supported observed frames still have valid boxes; box coordinates change substantially. |
| Q06 | 37.910062 -> 24.091490 | Identical reference, interval [5,11], 7 frame anchors and valid geometries. This is a coordinate-output degradation. |
| Q09 | 37.513171 -> 27.148417 | All 7 GT event positions remain in the decoded interval; an invalid box moves from frame 638 to frame 686, and other coordinates change. |
| Q11 | 87.980262 -> 0 | The only GT-supported observed frame loses its predicted box after the temporal shift. |
| Q03 | 47.189214 -> 63.313145 | Positive coordinate result retained; the same 3 GT event frames remain represented. |
| Q14 | 0 -> 20.425369 | A formerly omitted GT event frame becomes covered; positive support effect retained. |

Q06 is particularly informative. On the same seven valid frames, the mean
absolute normalized coordinate distance from the prediction to the saved SAM2
pseudo-box decreases from .070130707 to .032923863, while its saved GT sIoU
decreases by 13.818573 pp. This is evidence of greater agreement with the expert
pseudo-target accompanying worse task output. It is not a new expert-vs-GT IoU
evaluation, nor a proof that the spatial gradient alone caused all damage.

Q01 provides another concrete warning. Its spatial provider returns boxes whose
areas are 84.72%-98.33% of the entire image (mean 90.96%). At physical frame 80 the
native normalized box height moves .318 -> .807, and at frame 96 it moves
.318 -> .998. Mean coordinate distance to the expert over common valid frames
falls .374055558 -> .291255557 while sIoU falls. This supports investigating
coarse/incorrect pseudo-target attraction, rather than assuming the expert
always supplies a more accurate object target. No new image inspection or GT
box scoring was performed, so the exact object-identity error is not asserted.

## Implementation mechanisms that can transmit these errors

1. `scripts/desta_native_run_v3.py` constructs pseudo-targets on the original B1
   trace once and reuses that reference/interval/anchor support throughout latent
   updates. Final `decode_shared_reference_two_pass` freely generates temporal
   endpoints and then spatial anchors. In this arm 10/16 intervals change, while
   16/16 semantic references remain identical. Update-support versus final-output
   support mismatch is a real interface property; its isolated causal size is
   unmeasured.
2. The correction acts on the shared full merger F, entering both frozen B1
   readers and the identity residual in both native passes. Temporal and spatial
   gradient weights do not isolate their downstream effects.
3. `desta3d/native_adaptation.py` normalizes each full-F gradient, combines and
   projects them, then normalizes the combined step. For K1 a nonzero direction
   receives approximately the full rho=3% field radius. Expert confidence is
   not an absolute safety multiplier. A weak or incorrect pseudo-direction is
   therefore not automatically a small/no-op correction.
4. Targets are hard nearest-observation endpoint labels and quantized box
   coordinates. The current objective has no demonstrated safeguard for the
   existing correct endpoint/box. All these are code facts, not separately
   measured causal interventions.

## What the existing factorial says, without a new sweep

These are balanced marginal means over the already completed 27 configurations.
They are descriptive results on exposed Dev16, not proposed new settings.

| Factor | Values | Marginal sIoU (%) | Marginal vIoU (%) |
|---|---|---|---|
| Radius | .03 / .07 / .135 | 29.5579 / 28.4712 / 26.9548 | 15.7691 / 14.3604 / 12.3362 |
| Temporal weight (spatial=1) | .5 / 1 / 2 | 27.0743 / 27.8044 / 30.1053 | 14.2635 / 13.6612 / 14.5410 |
| Steps | 1 / 3 / 5 | 27.7249 / 28.5028 / 28.7563 | 15.6845 / 14.1609 / 12.6203 |

Larger radius worsens both means here. More steps worsen vIoU while sIoU slightly
increases, so “more steps make every box worse” is unsupported. Increasing the
relative temporal weight improves the marginal spatial mean, consistent with
coupled decisions and a problematic spatial pseudo-signal, but it does not prove
that temporal pseudo-targets are reliable. One configuration, K5/.03/T2, has
sIoU 37.034678% above B1 while vIoU is only 16.475637%; choosing solely by sIoU
would obscure substantial tube harm. All 27 vIoU means remain below B1.

## Checked alternatives and remaining limits

- All 16 before/after outputs have accepted format; the total number of invalid
  boxes is 2 before and 2 after, though their frame locations can change. No
  semantic-reference drift was found in this selected arm. A global parser
  failure or changed referent text does not explain its mean decline.
- Prior complete-stage scalar/tensor scoring and NumPy update reconstruction
  passed. This review verifies the original hashes for report/config/QC and all
  consumed prediction/step/expert files, and does not rescore GT.
- From one sealed C0 STEP01 per query, exact QC-column-space projection energy
  fraction has median 39.5470% for temporal (16 supported) and 31.7504% for spatial
  (15 supported); norm-fraction medians are 62.8864% / 56.3475%. Q10 has no spatial
  pseudo-support and stays separate. Compute energy as
  `sum((p @ inv(QC.T @ QC))*p)`, where `p=g@QC`. NumPy versus Torch solve differs
  at most 7.11e-15. These are initial-gradient results, not full update-trajectory
  statistics. They do not support saying the pseudo-gradient was almost entirely
  erased by projection; preserved energy also does not establish correctness.
- The full Dev64 and T-only/S-only ablations are incomplete/unrun. Their absence
  prevents branch-by-branch causal attribution. Single-seed, exposed development
  selection and a 16-parent panel limit generalization.

The strongest current explanation is **harmful pseudo-target attraction plus
shared finite correction, expressed both as temporal loss of box support and as
coordinate degradation on preserved support**. Temporal support loss is exactly
verified for the largest sIoU outlier; the upstream causes and their relative
shares are not uniquely identified. Sparse expert inputs and hard pseudo-targets
remain candidate contributors, not established universal failure mechanisms.

No scientific code, live configuration, experiment state, pins, automation or
production method was changed. Offline direction training, external inference,
Dev64, ablations and all new experiments remain stopped.

## Evidence and reproduction

- CPU readback: `scripts/analyze_desta_native_siou_readonly.py`.
- Local-only full case mapping and hashes:
  `artifacts/desta3d_v3/latent_oracle_v1/dual_expert_native_v1/siou_readonly_diagnosis_v1/{ROOT_READBACK.json,INPUT_HASHES.json}`.
- Original scores: `dual_expert_native_v1/scores/dev16/REPORT.json`.
- Score definition: `scripts/score_desta3d_v2_reference_audit.py`,
  `score_tube_independently`.
- Pseudo-target/update scope: `scripts/desta_native_run_v3.py`,
  `desta3d/native_adaptation.py`.
- Shared correction/final decoding:
  `vg_tta/desta3d_v3_decomposition.py`,
  `vg_tta/desta3d_v2_shared_reference_cached.py`.

Case Q01-Q16 follows the registered Dev16 order. No private source identifiers,
video/query text, GT box arrays, weights or raw tensors are included here.
