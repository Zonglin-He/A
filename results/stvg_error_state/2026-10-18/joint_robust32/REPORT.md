# Multi-candidate joint robust correction: fixed development32

## Finding

C improves HC relative to Top1 mean and Top1 robust estimation, but loses VidSTG gains. C has positive mean full-vIoU and unobserved actual-frame sIoU changes in both domains; it nevertheless fails the locked two-domain superiority comparison against A/B. Stop this fixed geometry-only candidate estimator; no threshold/kernel sweep, new model, confirmation or dense OPD is launched. D is an explanatory size intervention, not a selected method.

## Frozen setup and actual computation

Same16 exposed HC and16 exposed Vid parents; original saved query-start Before state primary, Source sensitivity separate; final-distribution MBR, original planned4 positions, original admitted DINO frame set, normalized center/log-size chart, four-dimensional constant correction and whole-trajectory application fixed. This is an output-layer experiment, not new Student parameter adaptation or online stream replay.

A exactly replays old DINO_full mean correction. B uses identical Top1 references with truncatedLS. C uses the original retained <=3 candidate boxes in each of the SAME admitted frames plus an outlier option. D shares C candidate choices, outlier assignments and full fitted delta, but applies only center components; original sizes stay bit exact. No new frame admission, target-score reranking, NMS, raw900 pool expansion or target-GT availability filter. There are56/50 admitted HC/Vid frames with168/144 retained candidate boxes; all106 original Top1 boxes reproduce exactly and separately match the old-probe recapture. The recapture is not substituted as a new expert output.

Source calibration reads only existing source-train36/domain caches and exact GT at original planned positions. Source-HC35 queries/113 positions and source-Vid22 queries/76 positions have>=2 known anchors; remaining queries cannot estimate within-query variation and are reported. Source-validation is unused. Each target uses the opposite source-domain calibration matching its Student checkpoint. The source/target parent sets are disjoint. Coordinate scales are parent-balanced within-query true-correction RMS; the cutoff is its parent-balanced95th percentile of normalized squared residuals. This is an explicit additional source-data requirement and engineering convention, not a target-tuned threshold or calibrated probability. See ACTUAL_PROTOCOL for all numeric scales/cutoffs and checkpoint hashes.

The exact global truncatedLS optimum is enumerated over at most4^4 candidate/outlier assignments. There is one vote per frame, no penalty for a correction far from the Student, no minimum-support whole-query rejection gate, and no newly fitted network. Single-frame-supported solutions are retained. All32 outputs for A/B/C/D and both bases were sealed before new target-GT scoring or old-label analysis.

## Primary results: differences against original Before (pp)

| Arm | HC full ΔvIoU [95% CI] | HC unobserved ΔsIoU [95% CI] | Vid full ΔvIoU [95% CI] | Vid unobserved ΔsIoU [95% CI] |
|---|---:|---:|---:|---:|
| A_mean | +0.689673 [-7.435740, +10.447267] | -2.675767 [-16.059301, +10.133904] | +1.931277 [+0.646712, +3.470272] | +6.176844 [-0.629624, +15.290373] |
| B_robust_top1 | +0.114785 [-8.514923, +10.041425] | -3.957152 [-17.636970, +9.283066] | +1.160544 [-0.286400, +2.699898] | +5.571481 [-0.996655, +14.272288] |
| C_joint | +2.902905 [-2.486872, +8.905767] | +1.566093 [-8.995164, +10.818232] | +0.713212 [-0.437848, +1.850524] | +2.642743 [-1.448153, +7.274857] |
| D_C_center | +2.675869 [-2.402710, +8.033716] | +2.074503 [-8.421386, +10.844662] | +1.058173 [+0.374377, +1.877085] | +2.810908 [+0.257163, +5.713973] |

All16 parents/domain remain, including Vid two no-reference no-ops. Unobserved means actual Decoder frames with exact GT, excluding all four planned positions. Missing GT stays unknown. Full s/t/R03/R05, fixed-time equality, within-MBR/interpolated supports, Source sensitivity and common-scale strata are in RESULTS.

| Contrast | HC Δv / unobserved Δs pp | Vid Δv / unobserved Δs pp |
|---|---:|---:|
| C_joint − A_mean | +2.213232 / +4.241860 | -1.218065 / -3.534101 |
| C_joint − B_robust_top1 | +2.788120 / +5.523244 | -0.447332 / -2.928738 |
| D_C_center − C_joint | -0.227037 / +0.508410 | +0.344961 / +0.168165 |

D improves unobserved means slightly in both domains, but HC full-v is lower than C. Source sensitivity reverses some D−C signs. These results do not justify disabling sizes or choosing a different variant by domain.

## Is the retained pool useful, and can geometry choose it?

| Domain | Known original admitted positions | Pool has IoU>=.5 | Original Top1 IoU>=.5 | C selected IoU>=.5 | Available good but not selected | No good retained candidate |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | 43 | 30 | 20 | 22 | 8 | 13 |
| vidstg | 24 | 22 | 11 | 8 | 14 | 2 |

These counts use only exact-GT anchors AFTER output seal and are geometry, not identity labels. Pool coverage is30/43 HC and22/24 Vid; C gets22 and8 good known references, while Top1 has20 and11. Thus candidates exist in many locations but geometry-only association still fails to select them, particularly in Vid. The absent-candidate statement is only about the original retained <=3 pool, not all raw DINO detections. Original Top1/best-retained/C-selected-conditional mean IoU is.454628/.658595/.499967 HC and.423859/.790515/.363094 Vid; Vid C abstains on3 known positions, so conditional selected means use a different denominator. Fixed-support abstention-zero means are separately saved.

C never increases the truncated geometric objective relative to B on any query. However, Vid actual tube gains are lower. This directly shows that a better geometric-consistency objective is not a certificate of target correctness. Old model-assisted WI/CP/MX/CI labels are used only for the ORIGINAL Top1; six old role-unknown queries remain UN. Alternative candidates have no new identity labels, and IoU does not certify the participant or role.

## Protection, active correction and negative tails

HC C rejects0 of56 admitted frames and makes0 query no-ops; it changes28 candidate choices,26 below the original per-winner.35 score. HC improvement is therefore not produced by simply refusing to update, but low-score alternatives remain a material risk. C improves12/16 HC queries over Base, reduces<-5pp full-v harms from5(A) to2, and rescues4 A-negative queries to nonnegative full-v. It also introduces severe failure H011: A+1.054pp full-v/+1.425pp unobserved becomes C−22.536/−59.327. Without new pixel review this is a geometric failure, not a newly certified identity error.

HC H002 moves A−26.171/−34.112 to C+6.818/+8.074, and H010−14.197/−8.286 to+2.017/+1.692. H005 largely recovers but still has−1.244pp unobserved loss. H009 gets worse. All cases remain in the table; no query was routed by GT or old labels.

Vid C rejects9 of50 frames and has the same2 no-reference query no-ops as A. It changes3 candidate choices; the remaining difference from A also includes robust rejection/averaging of Top1 frames. It rescues one A-negative query but creates a new<-5pp full-v failure (V007). V010 loses much of a good A correction: full-v gain+8.932 ->+1.255pp, unobserved+57.588 ->+7.921pp. B alone is also below A, so top1 robustness did not establish a simpler two-domain improvement.

C HC saved-score leave-one full-v mean range is[+.665227,+4.598835]pp, but unobserved range[−.995676,+5.625624]pp; the positive HC unobserved mean is not robust to every single-parent omission. The primary HC C bootstrap intervals cross0. All severe tails, positive/negative sums, parent signs, source sensitivity and small strata are retained. Intervals describe a repeatedly exposed32-parent development panel and fixed inherited trajectories, not independent confirmation or altered-stream replay.

## Reliability quantities and an audit clarification

Support count, candidate/outlier choices and leave-one normalized-delta distances were saved without GT. HC C has11 four-frame,2 three-frame and3 two-frame supports; Vid includes one single-frame support, which is inherently ambiguous. The sealed runner also saved the nearest distinct enumerated proposal gap. Auditing identified that such a proposal need not itself be a stationary association explanation. Original fields, predictions and metrics are preserved; DIAGNOSTICS additionally enumerates distinct self-consistent candidate/outlier modes and records their objective gap, normalized delta separation and direction cosine. Neither diagnostic was used as a selection/rejection threshold, and neither is called a correctness probability.

Independent auditing checks256 assignment/mean games and11800 conditional assignments, 848 leave-one checks, 448 output/identity reconstructions (maximum box error0.0), 1280 metric/actual-frame checks and1122 quality/aggregate/CI checks. Source calibration is independently recomputed. A is bit-exact to old outputs; D shares C selection and changes no size bits. All old inputs, code pins and production registries remain unchanged.

## Decision and scope

C satisfies positive two-domain mean changes against Base, but fails two-domain improvement against A and B. This fixed source-calibrated geometry-only estimator does not advance to independent confirmation or dense OPD. The experiment supports a limited HC candidate-selection opportunity and exposes unhandled wrong-consensus/constant-correction failures; it does not validate geometric consistency as a semantic association criterion. No source-scale, cutoff, pool or robust-kernel search follows, and no production method changes.

New GPU/model/expert/training/Student-update/visual-label counts are0. Cached CPU generation took1.215090s and scoring3.320073s, excluding original captures, calibration preflight and independent audit. These are not fresh deployment end-to-end costs.

[Full anonymous results](RESULTS.json) · [Decision](DECISION.json) · [Coverage and true competing modes](DIAGNOSTICS.json) · [Actual protocol](ACTUAL_PROTOCOL.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Aggregate figure](FOUR_ARMS.svg)

Raw GT coordinates, original query/parent text, media, weights, raw candidate tensors and correction vectors are excluded from the public export.
