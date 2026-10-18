# Strong sparse-anchor correction versus frozen-prior propagation

This is a source-only oracle diagnostic on the same12 HC and12 Vid validation queries, already exposed to auxiliary-prior development. Repaired HC30/Vid20 and cached input conditions remain frozen. It is not a deployable TTA result or Direct-versus-OPD comparison. No new training, expert calls, target inference or Student update.

## Four controls

N: Native represented by the original32 event-relative knots and decoded with the original interpolation. A: minimal probability-space knot correction satisfying the available sparse GT anchors, with that same interpolation. P: unanchored frozen-prior propagation from N. AP: propagation from the identical A start, with the identical sparse anchors reimposed on every clean estimate and proposal. Native time is exact in every state. Empty-anchor A=N/AP=P exactly.

Only existing knot columns touched by anchor interpolation may change during projection; all other logits remain byte exact. If needed, bounded convex SLSQP enforces finite sigmoid coordinates, with no unobserved GT input. The GT observation count is37 HC and23 Vid at the original four planned physical locations, not4 fabricated labels/query. Three Vid queries have no available oracle anchor and remain in the full denominator.

P and AP use the same old16-step .213 to .03 VE algebra, with NO new global noise. This intentionally separates deterministic propagation from artificial noise recovery. It is not standard diffusion sampling or a calibrated posterior; sigma is the frozen model conditioning input. No schedule/solver/strength tuning is performed.

## Effective anchor positive control

Query-mean values over nonempty anchors (HC12/Vid9). RMSE uses normalized cxcywh, not pixels. All predicted A/AP anchor coordinates are checked against sparse GT at full decoder output, not just latent knots.

| Arm | HC anchor RMSE | HC anchor IoU | Vid anchor RMSE | Vid anchor IoU |
|---|---:|---:|---:|---:|
| N | 0.045628236 | 0.635848229 | 0.105720699 | 0.518292894 |
| A | 0.000000014 | 0.999999882 | 0.000000018 | 0.999999516 |
| P | 0.041889764 | 0.677520218 | 0.101760236 | 0.527302899 |
| AP | 0.000000013 | 0.999999905 | 0.000000021 | 0.999999766 |

## Overall fixed-time results

Percent; full12-query denominator per domain. All tIoU values must equal Native exactly.

| Arm | HC vIoU | HC tIoU | HC sIoU | Vid vIoU | Vid tIoU | Vid sIoU |
|---|---:|---:|---:|---:|---:|---:|
| Native_raw | 43.280488 | 67.739714 | 64.016598 | 18.195709 | 35.521063 | 50.654046 |
| N | 43.367793 | 67.739714 | 64.120651 | 18.224082 | 35.521063 | 50.788860 |
| A | 47.879612 | 67.739714 | 69.970128 | 21.030601 | 35.521063 | 55.505562 |
| P | 46.434725 | 67.739714 | 67.952128 | 18.664347 | 35.521063 | 51.522236 |
| AP | 50.557837 | 67.739714 | 73.353574 | 21.395739 | 35.521063 | 56.076248 |

## Spatial supports

All supports are fixed from original physical Native time, original decoder frame IDs, original four planned anchors and annotation availability. Unobserved-native excludes all four planned positions. Outside-footprint-native further excludes every actual decoder frame whose interpolation touches an anchor-corrected knot. A is exactly N on this last support by construction. Unknown/empty supports remain null; counts are queries contributing to each support mean. Benchmark interpolation frames are reported separately.

| Arm | HC unobserved native sIoU (n) | HC outside footprint sIoU (n) | HC interpolated sIoU (n) | Vid unobserved native sIoU (n) | Vid outside footprint sIoU (n) | Vid interpolated sIoU (n) |
|---|---:|---:|---:|---:|---:|---:|
| N | 64.105503 (12) | 63.901562 (12) | 64.023480 (12) | 52.989321 (10) | 49.391321 (9) | 53.007050 (10) |
| A | 66.914053 (12) | 63.901562 (12) | 70.631985 (12) | 57.203902 (10) | 49.391321 (9) | 59.877025 (10) |
| P | 68.531220 (12) | 68.394570 (12) | 68.556224 (12) | 53.907743 (10) | 50.237686 (9) | 53.850054 (10) |
| AP | 71.218495 (12) | 68.600574 (12) | 74.665685 (12) | 57.980638 (10) | 50.252704 (9) | 60.529322 (10) |

## Matched propagation effects

Source-query paired Student-t95% intervals and positive/negative/neutral counts, percentage points. They describe this source-development cohort; no target confirmation or selected best path.

| Contrast / metric | HC | Vid |
|---|---|---|
| A - N / v | +4.511819 [+3.400408, +5.623230] 12/0/0 | +2.806519 [+0.421457, +5.191581] 9/0/3 |
| A - N / unobserved_native_s | +2.808549 [+1.855294, +3.761804] 12/0/0 | +4.214581 [+0.480946, +7.948217] 8/0/2 |
| A - N / outside_footprint_native_s | +0.000000 [+0.000000, +0.000000] 0/0/12 | +0.000000 [+0.000000, +0.000000] 0/0/9 |
| P - N / v | +3.066932 [+1.471408, +4.662456] 12/0/0 | +0.440265 [-0.294461, +1.174992] 8/2/2 |
| P - N / unobserved_native_s | +4.425717 [+2.157449, +6.693986] 12/0/0 | +0.918423 [-0.829073, +2.665918] 8/2/0 |
| P - N / outside_footprint_native_s | +4.493007 [+2.223599, +6.762416] 12/0/0 | +0.846366 [-1.250834, +2.943566] 6/3/0 |
| AP - A / v | +2.678225 [+1.218844, +4.137605] 12/0/0 | +0.365138 [-0.217613, +0.947889] 7/3/2 |
| AP - A / unobserved_native_s | +4.304442 [+1.940159, +6.668726] 12/0/0 | +0.776736 [-0.889204, +2.442676] 8/2/0 |
| AP - A / outside_footprint_native_s | +4.699011 [+2.423682, +6.974341] 12/0/0 | +0.861383 [-1.202867, +2.925634] 6/3/0 |
| AP - P / v | +4.123112 [+3.085109, +5.161116] 12/0/0 | +2.731392 [+0.463999, +4.998785] 9/0/3 |
| AP - P / unobserved_native_s | +2.687275 [+1.747171, +3.627378] 12/0/0 | +4.072894 [+0.486701, +7.659088] 9/0/1 |
| AP - P / outside_footprint_native_s | +0.206004 [+0.137690, +0.274318] 12/0/0 | +0.015017 [-0.070819, +0.100854] 6/2/1 |

The difference of differences (AP-P)-(A-N) is reported separately in RESULTS.json. It measures conditional responsiveness relative to the priors own unanchored drift; it does not by itself prove useful propagation, which requires AP>A on unobserved actual frames. Repeated anchoring can also change a nonlinear projection trajectory, so the interaction is not pure proof of learned semantic dependence.

## Decision and limits

cached_DINO_comparison_eligible
. Prelocked DINO eligibility requires positive AP-A mean on BOTH unobserved-native and outside-footprint-native supports in BOTH domains, after anchor qualification. It is an engineering continuation decision, not proof from statistical significance.

Independent checks: 1045. Cached GPU reservation: 1.733654s; cumulative 58.914447s under original14400s. CPU projections, full metric audit and upstream video/expert/feature work are separate, not online E2E latency.

This test gives observations explicit leverage on representable anchors. If AP fails to improve over A, weak anchor utilization no longer explains that comparison; the failure applies to this frozen propagation procedure and source cohort. It does not show that properly trained conditional imputation or every diffusion design cannot work. No learned association/TopK/fixed-video identity representation, new training or dense OPD was introduced. Actual pixel review is engineering only, not semantic identity annotation.

## Completed follow-up and final interpretation

The qualified cached-DINO follow-up is complete; see [DINO_REPORT.md](DINO_REPORT.md). All24 source queries retained; original learned association/Student/main method remain unchanged.

The strong-anchor positive control succeeds: A and AP source-oracle anchor IoU are approximately1 at the same positions and decoder coordinate error below2e-6. Relative to A, AP improves unobserved actual decoder sIoU by +4.3044 HC / +.7767 Vid percentage points, and outside-footprint sIoU by +4.6990 / +.8614 pp. These are not merely benchmark interpolation gains, though Vid intervals are wide.

However, unanchored P already improves unobserved actual decoder sIoU by +4.4257 / +.9184 pp. The conditional interaction (AP-P)-(A-N) is -.1213 / -.1417 pp on that support and -.3887 / -.0751 pp vIoU. Thus positive total propagation over A does not establish an additional sparse-anchor-conditioned correction mechanism.

On identical common positions, replacing actual DINO values by source GT improves AP vIoU by +4.0257 / +2.7883 pp; outside-footprint actual-frame sIoU changes only +.0345 / -.2782 pp. Reference-quality effects are concentrated in directly constrained and nearby interpolation regions in this cohort. This does not prove semantic identity error as the cause, nor that every nonlocal failure is propagation.

With all real DINO anchors, AP vIoU is46.0827 /18.3685 versus unanchored P46.4347 /18.6643: hard use of these observations partially offsets the unanchored priors gain. Hard anchoring is a diagnostic intervention, not a proposed replacement for uncertain observation likelihood.

Decision: preserve evidence that the frozen model can improve unobserved frames under this deterministic probe, but do not promote it to sparse-conditioned completion/dense OPD or target TTA. Correct-anchor utilization and total frozen propagation have now been separated; robust observation-conditioned extra propagation and unreliable-reference handling remain unestablished. No new training/model/target phase was started.

The v1 interpolation matrix used vector float32 physical-time arithmetic rather than the actual decoder scalar arithmetic; one DINO case exceeded unchanged2e-6 output tolerance. All v1 states/code/error receipts are preserved. v2 corrects this engineering mismatch, reuses all unanchored P exactly and recomputes anchor paths. Full output checks pass; no tolerance or science budget reset. See ENGINEERING_RECOVERY.json and RESOURCE_ACCOUNTING.json.
