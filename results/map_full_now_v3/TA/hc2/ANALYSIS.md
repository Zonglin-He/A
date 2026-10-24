# TA-STVG HC1902 — complete final-MAP and recomputed DP

All1,902queries/227parents have complete predictions, official scores and exact replay verification. Same upstream Source/eight observations/auxiliary trajectory/pooled features/posterior, kappa64/beta1/context1.5. Only the final q1 decoder is jointMAP, with the actual spatial DP solved on the new interval. This is a full fixed-protocol readout comparison, not a fresh unseen evaluation or a new hyperparameter search.

| Arm | tIoU (%) | vIoU (%) |
|---|---:|---:|
| Original Source: per-offset nativeMAP then physical hull |40.141475347240444|20.70467847062164|
| Source+MBR |48.64734502336978|24.99475237820813|
| SourceJointMAP: q0 jointMAP, original Source boxes |46.69329329573806|24.03909719476322|
| FullMBR |49.576841891153|27.218099277401947|
| MAP interval + old Full boxes |48.586627623587326|26.74719003117236|
| **FullMAP: MAP interval + recomputed DP** |**48.586627623587326**|**26.731488096319506**|

FullMAP−FullMBR: **−0.990214pp tIoU**, parent-cluster95%CI **[-1.267319,-0.695090]**; **−0.486611pp vIoU**, CI **[-0.645137,-0.322379]**. Relative to fixed-box MAP, the new spatial solve changes vIoU **−0.015702pp**, CI **[-0.050956,0.015933]**; t is identical by construction. Most of the mean loss in this comparison comes from the final interval decision, rather than the final spatial re-solve.

Against the same joint-MAP decoder applied to q0 with Source boxes, FullMAP adds **+1.893334pp t**, CI **[1.411589,2.383192]**, and **+2.692391pp v**, CI **[2.133693,3.275670]**. Against the original per-offsetMAP/hull Source, total deltas are **+8.445152t/+6.026810v**; these include the changed physical posterior/readout as well as feedback and space, not exclusively feedback effects. Against Source+MBR, FullMAP t is **−0.060717pp**, CI across0, while v is **+1.736736pp**, CI **[1.187044,2.323869]**. JointMAP is not relabeled as the untouched original Source readout.

## Negative tails and event duration

Compared with FullMBR, FullMAP has374queries with t loss>5pp,158with v loss>5pp, **377union cases** counted once. v improves594queries, declines877and ties431. Against original Source there are267temporal/264v/396union severe cases; versus q0JointMAP Source296/291/460. Against fixed-box MAP the spatial re-solve alone has4v severe losses and0temporal severe losses. Every signed outcome is retained.

Short events544: MAP−MBR t **+0.089618pp**, CI[-0.440245,0.638498], v **+0.168292pp**, CI[-0.148280,0.494526]. Long events1358: t **−1.422783pp**, CI[-1.728024,-1.108790], v **−0.748958pp**, CI[-0.930527,-0.567454]. Losses therefore concentrate in the long-event stratum in this full-set comparison. These are pointwise parent bootstrap intervals, no familywise correction, not a license to select different decoders by GT duration.

## Verification and costs

Whole domain sealed before labels. **11,412 official/dense checks**, maximum discrepancy **1.3322676295501878e-15**. Parent FullMBR and Source+MBR metrics match exactly. The postseal audit rechecked **9,510 input/prediction/receipt SHA bindings** and recomputed/replayed every output array exactly; means are recomputed from all rows. Original scoring/reporting code objects are reused unchanged, with separate output/readiness bindings recorded privately. Zero new Source/prefix/expert/RGB/GPU/optimizer work.

Prediction worker PID3080611/start239068052:1,902queries; summed conditional per-query CPU-stage wall **75.67115242686123s** (not total study time; excludes outer GC and stage setup). Prediction stage wall and audit worker details remain in private receipts. Scoring/audit run separately. Shared-host pauses and low-priority execution do not establish deployment latency or a standalone speedup.

All eight bridge cases were verified before the parallel jobs. Initial CPU scheduling used four allowed cores/nice15/idle-I/O; a precautionary own-MAP-only pause and stricter two-core/nice19 limit protect the ongoing original CG-Vid. See [runtime resource protection](../../RESOURCE_PRIORITY_UPDATE.md). Original Source/teacher acquisition provenance is retained, not charged again and not represented as zero deployment cost.

[Full means and paired CIs](report.json), [all signed values](all_signed.tsv), [all baseline severe tails/timegroups/rescues](all_signed_severe_timegroups.tsv), [official score receipt](score_receipt.json), [exact replay receipt](reporting_receipt.json). No captions, media, GT coordinates, raw features, masks or weights are public. Other MAP domains and original CG-Vid continue; this completion does not permit early strict-deletion GPU work.
