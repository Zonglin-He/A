# Official Tent/SAR STVG adapters: hc2

**Complete scored and replay-audited fixed panel:**128queries/128parents. Both methods use the previously selected development configurations, genuine episodic original temporal-decoder LayerNorm-affine updates, exact cached native decoder contexts and Source boxes, five iterations, and pinned unmodified official algorithm source. No new Source/expert/RGB is collected. Original queryreset/native-logit reset and all512configuration readouts were audited. This is a disclosed STVG task adapter, not canonical BN/fullnetwork/continual adaptation or a universal TTA ranking.

| Configuration | tIoU % | vIoU % | Severe t/v/union vs native Source |
|---|---:|---:|---|
| FullMAP | 52.319705 | 30.431495 | 17/20/30 |
| SourceJointMAP | 51.121909 | 28.063298 | 17/8/17 |
| Source | 46.509478 | 25.124253 | 0/0/0 |
| TentNative | 46.509478 | 25.124253 | 0/0/0 |
| TentJointMAP | 51.121909 | 28.063298 | 17/8/17 |
| SARNative | 46.509478 | 25.124253 | 0/0/0 |
| SARJointMAP | 51.121909 | 28.063298 | 17/8/17 |

Both adapted native intervals are unchanged on every query. The separately computed same-logit joint-MAP outputs are also unchanged from SourceJointMAP. This is a zero change in decoded accuracy under these frozen settings, not evidence that no optimization was executed.

| Method | Nonzero final parameter changes | Changed saved logits | Changed native intervals | Changed jointMAP intervals |
|---|---:|---:|---:|---:|
| Tent | 128/128 | 128/128 | 0/128 | 0/128 |
| SAR | 116/128 | 116/128 | 0/128 | 0/128 |

Exact saved parameter norms, logit magnitudes, per-query cost/reset diagnostics and raw reliability-filter counts are in [adaptation diagnostics](adaptation_diagnostics.json). SAR can leave no final parameter change when the official reliable-entropy filters admit no entries; no fallback output is inserted by this study. `SAR_empty_second_after_first_filter_iterations` uses the original first-filter membership intersected with the second-pass threshold, rather than treating the recorded raw second-pass count as its actual retained set. All original traces remain private and sealed.

| Signed comparison | Δt pp [95% CI] | Δv pp [95% CI] |
|---|---:|---:|
| TentNative-Source | +0.000000 [0.000000, 0.000000] | +0.000000 [0.000000, 0.000000] |
| TentJointMAP-Source | +4.612431 [2.709373, 6.582400] | +2.939045 [1.824072, 4.120582] |
| SARNative-Source | +0.000000 [0.000000, 0.000000] | +0.000000 [0.000000, 0.000000] |
| SARJointMAP-Source | +4.612431 [2.709373, 6.582400] | +2.939045 [1.824072, 4.120582] |
| FullMAP-Source | +5.810228 [3.418946, 8.206662] | +5.307242 [2.696017, 7.980195] |
| FullMAP-SourceJointMAP | +1.197797 [-0.103761, 2.578902] | +2.368197 [-0.073230, 4.878119] |
| FullMAP-TentNative | +5.810228 [3.418946, 8.206662] | +5.307242 [2.696017, 7.980195] |
| FullMAP-TentJointMAP | +1.197797 [-0.103761, 2.578902] | +2.368197 [-0.073230, 4.878119] |
| FullMAP-SARNative | +5.810228 [3.418946, 8.206662] | +5.307242 [2.696017, 7.980195] |
| FullMAP-SARJointMAP | +1.197797 [-0.103761, 2.578902] | +2.368197 [-0.073230, 4.878119] |

All comparisons are on the same previously exposed128-parent panel, not the complete target set. Pointwise parent-cluster intervals are not familywise guarantees. In Vid, the FullMAP−native-source/TTA vIoU interval crosses zero; do not claim a significant advantage in both domains. FullMAP uses eight expert observations and zero optimizer steps; Tent/SAR use zero expert observations and five iterations. These are different budgets, explicitly disclosed.

All signed per-query metrics/severe losses/rescues/duration groups and observed/unobserved tradeoffs are retained in report.json, all_signed.tsv and all_signed_severe_timegroups.tsv. Native Source uses per-offset MAP plus physical hull; SourceJointMAP isolates readout effects. FullMAP is the already-completed current MAP configuration on the identical cohort, not a newly selected model.

Actual cached-context adaptation costs: {"Tent": {"queries": 128, "adaptation_seconds": 15.499139379244298, "decoder_calls": 1536}, "SAR": {"queries": 128, "adaptation_seconds": 25.544279270339757, "decoder_calls": 2816}}. These include real decoder forward/backward and optimizer work, but exclude inherited Source/context acquisition; they are not complete deployment timings or free acquisition. Actual deployment Source/currentFull timings are reported independently. Whole256 predictions were sealed before independent official scoring (1,280dense checks), then768prediction/logit/receipt SHA checks, all adapted native/joint readout replays, allmeans and episodic resets passed. Original parameter/evidence settings are unchanged, with no new sweep based on these results.