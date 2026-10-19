# ReferDINO student-conditioned feedback: HC128

Completed interface probe; this is **not** a full OPD result. All128 random parent predictions sealed before new GT scoring. One frozen ReferDINO checkpoint, identical8 physical frames and original full query. Original E+finalMBR time fixed. No new expert or trained model; only per-video temporary content states updated.

**Measured decision:** retain a small fixed-feedback transfer signal, but do **not** enter full OPD for this initial-reference configuration. S1 improves full vIoU +0.5181pp [0.0169,1.0908] and outside8 actual-node sIoU +1.2611pp [0.2065,2.5246]. Refresh minus cached-old feedback is only +0.0107pp full-v and +0.0130pp outside8, both intervals cross zero. This does not establish a2–3pp trajectory gain or a useful refresh advantage. It is a conclusion about this configuration, not all OPD.

## Configuration and data

The existing official HC dev split has historical Source/full-method evaluation exposure. Exclude NH16 engineering parents; uniform parent shuffle seed2026101901, reserve32 before selecting128, one uniformly random query per parent. Reserved32 were neither run nor scored and are excluded from this interface selection; they are not globally never-exposed data. NH001 checks interface only. No success or reference-availability selection.

Intervene before spatial decoder layer0: lock the native winning tracked query’s initial proposal ancestor separately in each frame, replace exactly those initial references with student boxes. Preserve native content/text and899 other references; native six-layer spatial evidence reading/pruning/Hungarian tracking/temporal/mask decoders execute. Native sigmoid reference projection to [1e-5,1-1e-5] is reported. Trace every intervention; another selected native lineage is not called successful injected-query correction.

S1 is40 fixed Adam lr.01 updates, mean8[5L1+2(1-GIoU)], on eight independent256D content states through the existing frozen TA-STVG decoder. From identical S1, separately use old T(S0) versus refreshed T(S1),40 steps each, fresh identical Adam. No sampling/OPD claim, no parameter sweep or best-step selection. Width/height may change through the original decoder bbox head; the E baseline itself is immutable.

## Student results

All entries are parent-macro paired differences in percentage points;95% paired bootstrap intervals. The space endpoint includes known actual nodes outside all8 teacher positions, with unknown labels omitted rather than filled with zero. All parents remain in full-v means.

| Comparison | Full vIoU pp [95% CI] | Outside8 actual-node sIoU pp [95% CI] |
|---|---:|---:|
| S1_minus_E_MBR | +0.5181 [+0.0169, +1.0908] | +1.2611 [+0.2065, +2.5246] |
| Old_feedback_minus_E_MBR | +0.5347 [-0.0955, +1.2512] | +1.0312 [-0.3134, +2.5500] |
| Refreshed_feedback_minus_E_MBR | +0.5454 [-0.0836, +1.2672] | +1.0442 [-0.3009, +2.5680] |
| Refreshed_feedback_minus_Old_feedback | +0.0107 [-0.0122, +0.0454] | +0.0130 [-0.0131, +0.0391] |
| Refreshed_feedback_minus_S1 | +0.0273 [-0.3059, +0.4284] | -0.2168 [-0.8884, +0.5692] |
| Old_feedback_minus_S1 | +0.0166 [-0.3170, +0.4198] | -0.2298 [-0.8994, +0.5561] |

## Teacher quality and provenance

| Same known observation support | Actual selected-frame IoU pp [95% CI] |
|---|---:|
| Native_minus_E | +6.7646 [+2.7172, +11.2084] |
| T0_minus_E | +7.1433 [+3.1806, +11.4975] |
| T0_minus_Native | +0.3787 [-0.3858, +1.1500] |
| T1_minus_T0 | -0.0927 [-0.6142, +0.4074] |
| T1_minus_S1 | +5.4021 [+1.5737, +9.6165] |

IoU0.5 recovery/damage groups are descriptive postseal analysis, not online gates or identity labels. No independent identity annotation exists; low IoU or large motion is not sufficient to claim identity recovery.

- Native: known425, originally below0.5:159, recovered≥0.5:51; originally≥0.5:266, damaged<0.5:35.
- T0: known425, originally below0.5:159, recovered≥0.5:49; originally≥0.5:266, damaged<0.5:28.
- T1: known425, originally below0.5:156, recovered≥0.5:43; originally≥0.5:269, damaged<0.5:26.
- T0: intervention survives last spatial pruning at631/1024 slots; final chosen track actually contains it at468/1024. Projected reference elements:107.
- T1: intervention survives last spatial pruning at653/1024 slots; final chosen track actually contains it at502/1024. Projected reference elements:89.

### Which query produced the apparent corrections?

Among425 known observation positions, T0 recovers49 of159 E-bad boxes to IoU≥0.5 while damaging28 of266 E-good boxes below0.5. Of those49 recoveries, **48 select another final query lineage; only1 retains the intervened query**. For T1 versus S1,40 of43 recoveries select another lineage and3 retain the intervention. These are descriptive postseal categories; the T0 and T1 baseline-bad supports differ. The result cannot be described as49 successful corrections of the injected student query.

Changing one reference can influence other queries through native self-attention; selecting another lineage does not prove it is numerically identical to an untouched cached native box. Only262/1024 T0 boxes and250/1024 T1 boxes have IoU≥0.99 to some actual native candidate. Thus neither “every changed output is new corrective information” nor “every alternate lineage is an unchanged native candidate” is supported. The measured quality comparisons, lineage audit and student uptake must be read together.

## State coverage and tails

Real S1 changes128/128 queries and8320 actual nodes; mean state L2=12.654448, mean center displacement/E width-height=0.089016. T1 versus T0 differs by more than1e-6 in128/128 queries. No artificial box perturbations were substituted.

- S1: positive77, negative43, unchanged8; worse than−3pp:3. Worst/best5 retained in DECISION_PUBLIC.json and all signed rows.
- Old_feedback: positive74, negative46, unchanged8; worse than−3pp:6. Worst/best5 retained in DECISION_PUBLIC.json and all signed rows.
- Refreshed_feedback: positive72, negative48, unchanged8; worse than−3pp:6. Worst/best5 retained in DECISION_PUBLIC.json and all signed rows.

## Cost and verification

Total new GPU allocation including failed/preflight attempts=1516.196s. Source128 forwards; main Refer128 native+128 T0+128 T1 query forwards, each8 frames; NH001 engineering repeats counted separately. Student15,360 actual updates, three40-step stages per parent; separate decoder/conditional teacher per-query timing in COST_BREAKDOWN.json. Refresh uses an additional teacher query compared with cached-old uptake; equal student budgets do not imply equal expert computation. Old cache acquisition cost is excluded. No new DINO-T calls or pretrained weight updates.

Independent audit replays saved pruning/matching/native selection, exact S1 reference source, shared starting states, all saved Adam steps/losses, independent DenseTube full metrics/outside8 denominators and parent bootstrap intervals. It is not a second full GPU inference replication. Production CURRENT/MAIN/PAPER remain unchanged.

## Decision boundary

The experiment asks whether this specific initial-reference intervention provides reliable corrections and whether refreshed feedback helps actual student uptake. Teacher changes alone do not establish benefit. Alternate selected native queries cannot be attributed to correcting the injected query. A direct-supervision gain is not an OPD gain. No automatic full OPD, parameter sweep or held32 confirmation is triggered.

## Final interpretation and resource decision

Native ReferDINO has real local information beyond E (+6.7646pp on common observed GT support,127 contributing parents). Conditioning adds +0.3787pp relative to native with interval crossing zero; refreshing changes feedback on all128 videos but T1−T0 observed IoU is−0.0927pp with interval crossing zero. Therefore changed output is not evidence of a stronger corrective teacher.

S1 motion is sufficient to exercise the interface:122 videos have maximum actual-node center displacement greater than0.01 E-width/height,108 exceed0.05,103 exceed0.1; median per-video maximum is0.522869. The missing refresh gain cannot be explained solely by identically unchanged student boxes. This is not a claim that every relevant identity/state error was explored.

Full-v S1 has77 improvements/43 declines/8 unchanged; worst RC036−11.3769pp and RC126−6.6888pp, positive RC091+22.5889pp and RC097+15.5598pp. Both80-step branches have6 losses below−3pp versus3 for S1. Additional fitting lowers its own anchor objective, but does not establish improved unobserved trajectory quality. All tails remain in the main128 mean.

Conditional expert-stage latency (8-frame preprocessing/forward/postprocess, tracing included, excluding RGB decode/model loading): native mean0.209796s, T0 mean0.207395s, T1 mean0.211617s; medians0.189890/0.187440/0.191964s. End-to-end allocation includes repeated RGB decoding, Source capture,16,000 trajectory decoder calls/32,000 offset decoder calls, student backpropagation, loading and guards; it must not be replaced by these small expert-forward times.

Original E reconstructed independently on all128 from canonical cached admitted pools and frozen source calibration, maximum error0. Two original E cases have no admitted reference and remain included. All512 possible original-four input slots were accounted for:508 actual cached DINO image receipts match RGB; one original query had zero calls, and another had calls but no admitted references. New ReferDINO does not use the old question skip. Source endpoint logits were not retained in this probe; canonical finalMBR is computed during capture and exactly fixed across arms, without claiming an independent temporal-probability reconstruction.

**Close this finite probe without full OPD, additional observations, slot/parameter sweeps or reserved32 evaluation.** Preserve the original native teacher advantage and first-stage fixed-feedback transfer as separate evidence. No production method promotion.
