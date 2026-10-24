# CG-STVG HC full evaluation — complete and verified

All **1,902 queries / 227 parent videos** completed under the previously frozen HC configuration: gamma=0, unobserved share=0, link=0, beta=0.5, kappa=64, context=1.5. No full-result parameter selection was performed. Native420 FP32, cross-domain source-checkpoint EMA and the same ReferDINO eight observations are retained. This is the local cross-domain protocol, not a claim to reproduce the paper's in-domain accuracy.

| Displayed method | Mean tIoU (%) | Mean vIoU (%) |
|---|---:|---:|
| Source: original CG native MAP | 23.220480492816346 | 14.673931966268045 |
| Source + our MBR readout | 42.632981520545684 | 26.153691190229933 |
| Full | 44.89914441015363 | 27.33108741943422 |

| Paired comparison | tIoU delta (pp), parent-cluster 95% CI | vIoU delta (pp), parent-cluster 95% CI |
|---|---|---|
| MBR − original Source | +19.412501 [18.177353,20.670274] | +11.479759 [10.621062,12.365383] |
| Full − Source+MBR | +2.266163 [1.732572,2.806761] | +1.177396 [0.794005,1.564788] |
| Full − original Source | +21.678664 [20.265398,23.124163] | +12.657155 [11.644551,13.700790] |

The original nativeMAP-to-MBR readout contributes most of the total improvement. The matched-MBR comparison measures the additional effect of the retained feedback pipeline. MBR is our temporal-readout component, not behavior inherited from the untouched Source. Frozen internal keys remain SourceNative=displayed Source and Source=displayed Source+MBR; numerical files and old hashes are preserved. Full−Source is never substituted for Full−Source+MBR when attributing feedback gains.

## Losses and component scope

Against **Source+MBR**, Full has 348 temporal-severe and 242 vIoU-severe cases; their union is **377/1902** (t or v drops more than 5 percentage points, a case counted once). vIoU improves on 1,006 queries, declines on 723 and ties on 173. Observed-frame sIoU falls **1.164300pp**, CI **[-2.146002,-0.159625]**, on the 1,857 queries where that statistic is defined. All 1,902 remain in downstream t/v metrics.

Against the **original native Source**, Full has 134 temporal-severe / 111 vIoU-severe / **153 union**; Source+MBR has 127 / 103 / **137 union**. Average gains do not remove these failures. The separately named report retains both baselines and every signed outcome.

HC gamma=0 disables shared correction; unobserved share=0 and link=0 retain the original Source boxes on unobserved frames. Their fixed-support sIoU difference is exactly zero on all 1,902 queries. Observed-frame selection remains; Space at the matched prior interval changes vIoU by **-0.220303pp** versus Source+MBR. Full−Space adds **1.397700pp vIoU**, CI **[1.024474,1.778160]**. The result supports this conservative selected HC configuration, not effective full spatial propagation on CG.

For short events (544 queries, GT duration <=5s), Full−Source+MBR is t **+0.819149pp**, CI [0.063703,1.604174], and v **+0.314703pp**, CI [-0.251000,0.905600]. For long events (1,358 queries), t **+2.845821pp**, CI [2.175310,3.490116], and v **+1.522982pp**, CI [1.055819,1.984043]. All intervals are pointwise parent bootstrap intervals without familywise correction.

## Exposure and strata

All 227 HC parents previously participated in CG development/CV, one query per parent. The 227 selection queries yield Full−Source+MBR v **+1.384979pp**, CI [0.471152,2.409356]. The remaining 1,675 queries across 204 of those parents yield **+1.149264pp**, CI [0.761155,1.539380]. The remaining-query stratum is not an independent-parent test, and the full set is historically exposed through the wider TA work. These strata are reported rather than described as unseen confirmation.

## Verification and cost

All predictions sealed before independent CPU official labels. **9,510 official/dense checks**, maximum discrepancy **1.4432899320127035e-15**. The postseal audit rechecked all **5,706 prediction/compact/receipt SHA bindings** and replayed every saved output array exactly. Both native checkpoint no-op bridges passed for new HC acquisitions. No inference, new labels or expert calls were needed for these postseal reports.

Actual HC worker segment: PID1427705/start237101753. It processed 227 compatible legacy acquisitions and 1,675 new Source queries / 1,677 Source forwards including two native reference extras / 1,675 new same-expert8 joint calls. Stage wall **10034.061705 seconds** (~2h47m), sum query pipelines **9726.434512 seconds**; these include cache and research-audit work and are not uncached deployment latency. Source capture seconds **3658.799482**, feedback seconds **1057.909375**, expert forward seconds **317.535406**; these are nested pipeline components and are not added together. Source model load **8.892391 seconds** once for the actual worker; expert load **5.528526 seconds**. Legacy acquisition costs remain separate provenance, not zero deployment cost. Compact plus predictions total **3,806,995,585 private bytes**; old raw caches are unchanged.

The [main report](report.json), [all signed values](all_signed.tsv), [signed severe cases, groups and rescues](all_signed_severe_timegroups.tsv), [audit receipt](reporting_receipt.json), [original-Source comparison](named_source_comparison/report.json), [selection-query stratum](selection_queries/report.json) and [remaining-query stratum](remaining_queries/report.json) retain full-precision values. Media, captions, coordinates, raw features, masks and checkpoints remain private.

Vid full evaluation continues unchanged. The four-row TA strict-deletion queue remains unstarted until both domains and reports complete; this HC completion alone does not release its dependency gate.
