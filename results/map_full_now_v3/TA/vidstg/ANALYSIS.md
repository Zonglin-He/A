# TA-STVG VidSTG10303 — complete final-MAP and recomputed spatial DP

All 10,303 queries from 732 parent videos have complete predictions, independent official scores and exact postseal replay verification. Same frozen upstream Source, eight ReferDINO observations, shared geometry, auxiliary trajectory, pooled features and q0/q1; kappa64/beta1/context1.5. The final decision is q1 joint physical-interval MAP with C-order earliest ties, followed by the original spatial DP genuinely solved on the new interval. No new model, Source, prefix, expert, RGB, GPU or optimizer work. This full dataset was historically exposed; this is a fixed-protocol readout comparison, not an unseen test or a hyperparameter search.

| Arm | tIoU (%) | vIoU (%) |
|---|---:|---:|
| Original Source: per-offset native MAP then physical hull | 31.7284544051222 | 11.1384836730865 |
| Source+MBR | 33.8045661604283 | 11.7776551418415 |
| SourceJointMAP: q0 joint MAP, original Source boxes | 33.4508057889563 | 11.6416270781583 |
| FullMBR | 34.6043780655094 | 14.3833140568339 |
| MAP interval + old Full boxes | 34.6640674145293 | 14.4140925592616 |
| **FullMAP: MAP interval + recomputed DP** | 34.6640674145293 | 14.4336934652604 |

## Paired comparisons

| FullMAP minus baseline | tIoU delta pp [95% CI] | vIoU delta pp [95% CI] |
|---|---:|---:|
| Original Source: per-offset native MAP then physical hull | +2.935613 [+2.547106, +3.355527] | +3.295210 [+2.893241, +3.708519] |
| SourceJointMAP: q0 joint MAP, original Source boxes | +1.213262 [+0.946621, +1.495964] | +2.792066 [+2.420883, +3.168084] |
| Source+MBR | +0.859501 [+0.610736, +1.120017] | +2.656038 [+2.286744, +3.032796] |
| FullMBR | +0.059689 [-0.063341, +0.180905] | +0.050379 [-0.009046, +0.108138] |
| MAP interval + old Full boxes | +0.000000 [+0.000000, +0.000000] | +0.019601 [+0.009420, +0.030069] |

The overall MAP-minus-MBR tIoU and vIoU intervals both cross zero. The observed means are close, with no established MAP advantage or formal equivalence claim. SourceJointMAP uses the same decision rule on q0, whereas FullMAP applies it to revised q1 and solves the spatial path. Original TA Source uses per-offset MAP followed by a physical hull, so OriginalSource and SourceJointMAP must not be conflated. Joint-MAP Source alone adds +1.722351t/+0.503143v pp over OriginalSource; that readout gain is not credited to expert feedback. Relative to the matched joint-MAP Source, FullMAP adds +1.213262t/+2.792066v pp. Actual new-interval spatial solving adds a small +0.019601v pp versus fixed-box MAP, CI[+0.009420,+0.030069].

## Signed outcomes and severe tails

Severe means query tIoU or vIoU drops by more than5pp, union counted once; rescue means either increases by more than5pp, union counted once. A query may have a severe loss in one metric and a rescue in the other.

| Baseline | Severe t | Severe v | Severe union once | Rescue union once |
|---|---:|---:|---:|---:|
| OriginalSource |1139|1154|1934|4275|
| SourceJointMAP |970|1159|1880|3441|
| FullMBR |649|168|656|842|
| Fixed-box MAP |0|4|4|13|

Compared with FullMBR, MAP v improves2,892queries, declines3,296and ties4,115; compared with SourceJointMAP it improves4,980, declines2,891and ties2,432. Mean improvements are not universal per-query gains. Every positive, negative and unchanged result remains in the signed TSV files.

## Event duration

| Duration | Queries | Comparator | Delta t pp [95% CI] | Delta v pp [95% CI] |
|---|---:|---|---:|---:|
| short_le5s | 6060 | Source | -0.327366 [-0.591641,-0.052901] | +1.006086 [+0.677979,+1.362015] |
| short_le5s | 6060 | SourceJointMAP | +0.148318 [+0.004023,+0.295648] | +1.277337 [+0.955621,+1.625285] |
| short_le5s | 6060 | FullMBR | +0.029094 [-0.137705,+0.191013] | +0.006794 [-0.072695,+0.082868] |
| long_gt5s | 4243 | Source | +7.595912 [+6.906868,+8.334642] | +6.564616 [+5.889406,+7.278668] |
| long_gt5s | 4243 | SourceJointMAP | +2.734251 [+2.143682,+3.355731] | +4.955456 [+4.320505,+5.608105] |
| long_gt5s | 4243 | FullMBR | +0.103387 [-0.083743,+0.287940] | +0.112629 [+0.021624,+0.205785] |

Short-event temporal performance relative to OriginalSource declines -0.327366pp, CI[-0.591641,-0.052901], despite a positive overall temporal mean. Relative to the matched SourceJointMAP, short-event t is +0.148318pp, CI[+0.004023,+0.295648]. Long-event MAP-minus-MBR v is +0.112629pp, CI[+0.021624,+0.205785], while short-event v is +0.006794pp with CI crossing zero. These are descriptive fixed strata and pointwise parent-cluster bootstrap CIs without familywise correction; they do not authorize selecting decoders by GT duration.

## Verification and conditional research cost

All10,303 predictions sealed before fixed official labels. 61,818 official/dense checks, maximum discrepancy2.4424906541753444e-15. Inherited FullMBR/Source+MBR metrics match the parent exactly. The postseal report reverified51,515 parent/prediction/receipt SHA bindings, exactly replayed all MAP and MBR arrays and recomputed every reported mean. The score/report source code objects are inherited unchanged in independent processes with the new output/readiness bindings. Reporting completed successfully with exit0; no scoring or model was rerun for this analysis.

Prediction worker PID3080612/start239068052:10,303queries, summed conditional per-query wall797.8785783317871s. Whole prediction stage wall2562.443684342783s includes outer GC/setup/shared-host scheduling and pauses; scoring and final replay audit are separate stages. These are cache-only research costs, not uncached deployment latency or an end-to-end speedup. Initial four-core/nice15 scheduling was narrowed to CPU14-15/nice19/idleIO and one computational thread; only own MAP groups may pause, originalCG processes/priority/affinity remain unchanged. See [resource policy](../../RESOURCE_PRIORITY_UPDATE.md). Historical Source/expert acquisition remains separately bound and is not represented as zero deployment cost.

[All means and paired CIs](report.json), [all signed values](all_signed.tsv), [all baseline severe/timegroups/rescues](all_signed_severe_timegroups.tsv), [official score receipt](score_receipt.json), [exact replay receipt](reporting_receipt.json). No private captions, videos, GT coordinates, raw features, masks, weights or credentials are published. Original CG-Vid full and its committed-cache MAP consumer continue; strict-deletion GPU reconstruction remains deferred until all four MAP reports and originalCG audits/owned exit.
