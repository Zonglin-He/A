# CG-STVG HC1902 — complete final-MAP and recomputed DP

All1,902queries/227parents sealed, independently scored and exactly replayed. Frozen HC gamma0/u0/link0/beta0.5/kappa64/context1.5, original420FP32 cross-source EMA, same eight ReferDINO observations. No new Source, expert, raw-video, optimizer or GPU work. Reuse identical upstream posterior inputs; change final q1 readout to jointMAP and actually solve the retained spatial decisions at that new interval. Unobserved boxes still equal Source by this conservative HC configuration; no validation claim for full propagation.

| Arm | tIoU (%) | vIoU (%) |
|---|---:|---:|
| Original Source: native MAP |23.220480492816346|14.673931966268045|
| Source+MBR |42.632981520545684|26.15369119022993|
| SourceJointMAP: q0 MAP with Source boxes |23.220480492816346|14.673931966268045|
| FullMBR |44.89914441015363|27.33108741943422|
| MAP interval + old Full boxes |38.252644106663894|23.482252595184768|
| **FullMAP: MAP interval + recomputed DP** |**38.252644106663894**|**23.48261111234244**|

FullMAP−FullMBR: **−6.646500pp tIoU**, parent-cluster95%CI **[-7.412265,-5.903820]**; **−3.848476pp vIoU**, CI **[-4.344024,-3.364627]**. The new spatial solve versus fixed-box MAP changes vIoU only **+0.000359pp**, CI **[-0.000863,0.001683]**. The mean loss in this specific comparison is therefore predominantly due to the interval decision rather than the retained final spatial solve.

Against the original Source, which has identical reported metrics to q0JointMAP here, FullMAP adds **+15.032164pp t**, CI **[13.795293,16.300423]**, and **+8.808679pp v**, CI **[7.992068,9.678131]**. Against Source+MBR, FullMAP is **−4.380337pp t**, CI[-5.223131,-3.538441], and **−2.671080pp v**, CI[-3.176624,-2.148095]. The reference decoder must be named explicitly; a positive original-MAP Source gain does not negate the loss versus MBR.

## Severe tails and event groups

Relative to FullMBR:709temporal-severe/512v-severe/**713union** cases (loss>5pp; union counted once). v improves491queries, declines1139and ties272. Relative to original Source/q0JointMAP:102temporal/98v/**133union** severe cases. The original FullMBR had153original-Source-relative union cases, so MAP has20fewer severe cases under that reference while also having a lower mean. These two baseline comparisons are not interchangeable. The final spatial re-solve versus fixed-box MAP has0severe cases.

Short544: MAP−MBR t **−3.188980pp**, CI[-4.355131,-2.067416], v **−1.983857pp**, CI[-2.748546,-1.230539]. Long1358: t **−8.031545pp**, CI[-9.028389,-7.094994], v **−4.595422pp**, CI[-5.208297,-4.001449]. All are pointwise parent bootstrap intervals, without familywise correction. No duration-based decoder selection is performed.

## Validity, exposure and actual cost

**11,412 official/dense checks**, maximum discrepancy **1.4432899320127035e-15**; all MAP predictions sealed before official labels. Original Source and FullMBR metrics match parent results exactly. Postseal audit **9,510 input/prediction/receipt SHA checks**, every output array recomputed exactly and all means recomputed. Original scoring/reporting code objects are reused unchanged in separate processes with only output/readiness bindings.

All227HC parent videos previously participated in CG development/CV; all official sets have historical TA exposure. This is not an independent unseen target-domain confirmation. The profile and MAP rule were fixed before this evaluation; no post-result tuning, failure filtering, new teacher or decoder switching is authorized by this result.

Actual prediction worker3080613/start239068052:1,902queries; sum conditional per-query CPU-stage wall **163.73399950237945s**, including any pauses during a query but excluding outer GC/setup. This is not total study wall or deployment latency. The four initial CPU jobs were protected by own-group pause/resume and narrowed to CPU14–15/nice19/idle I/O; original CG-Vid is untouched. See [resource protection record](../../RESOURCE_PRIORITY_UPDATE.md). Legacy visual acquisition is provenance, not new work or zero deployment cost. Pause observations are not a randomized causal speed benchmark.

[Means and every paired CI](report.json), [all signed values](all_signed.tsv), [signed severe/timegroups/rescues](all_signed_severe_timegroups.tsv), [official receipt](score_receipt.json), [all-array/SHA replay receipt](reporting_receipt.json). Private captions/media/GT coordinates/features/masks/weights remain private. Both Vid MAP jobs and the original CG-Vid full continue; strict deletion GPU work is still deferred.
