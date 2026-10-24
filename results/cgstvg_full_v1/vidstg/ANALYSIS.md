# CG-STVG VidSTG full evaluation — complete and verified

All **10,303 queries / 732 parent videos** completed with the previously frozen Vid profile: gamma=.5, unobserved share=.5, link=.25, beta=1, kappa=64, context=1. Original native420 FP32, cross-domain source-checkpoint EMA, the same ReferDINO eight observations and both-anchor complete-box geometry/exact DP are retained. No parameters were chosen from these full results. This local cross-domain protocol is not an in-domain paper accuracy reproduction.

| Displayed method | Mean tIoU (%) | Mean vIoU (%) |
|---|---:|---:|
| Source: original CG native MAP | 17.318935 | 7.532313 |
| Source + our MBR readout | 23.481297 | 9.965719 |
| Full | 27.166958 | 11.494698 |

| Paired comparison | tIoU delta (pp), parent-cluster 95% CI | vIoU delta (pp), parent-cluster 95% CI |
|---|---|---|
| MBR − original Source | +6.162362 [5.607278, 6.737183] | +2.433405 [2.174980, 2.710493] |
| Full − Source+MBR | +3.685662 [3.179263, 4.244420] | +1.528979 [1.168330, 1.900103] |
| Full − original Source | +9.848023 [9.067188, 10.672444] | +3.962385 [3.513628, 4.435980] |

The MBR-only change contributes +6.162362t/+2.433405v pp. Full−original Source includes this decoding effect and must not all be credited to visual/expert feedback. Full−Source+MBR isolates the additional feedback under matched MBR. Frozen internal aliases remain SourceNative=displayed Source and Source=displayed Source+MBR.

## Losses and spatial scope

Against **Source+MBR**, Full has **1268 temporal-severe / 1329 vIoU-severe / 2024 union** cases. Severe means a drop greater than 5pp; a query in both metrics is counted once in the union.
Against **original native Source**, Full has **857 temporal-severe / 968 vIoU-severe / 1375 union** cases. Severe means a drop greater than 5pp; a query in both metrics is counted once in the union.

Matched-MBR vIoU improves on 4018 queries, declines on 3012 and ties on 3273. Mean improvements do not erase these tails.

Fixed-support unobserved-frame sIoU changes **-2.710290 [-3.274406, -2.159976]pp**, with the entire CI negative. Observed-frame sIoU is **+0.727891 [-0.037062, 1.477209]pp**, with a CI crossing zero. Full fixed-support sIoU is lower than Source. This is not evidence that spatial propagation improves localization overall. SharedCorrection vIoU is 9.483226%, Space 9.800345%, both below Source+MBR 9.965719%; Full−Space vIoU is +1.694353pp, CI[1.432625,1.979867]. These are frozen controls, not post-result configuration selection.

## Duration and exposure

Short events (<=5s), 6060 queries: matched-MBR tIoU +2.119711pp CI[1.6214377974797987, 2.6426024092146436]; vIoU +0.439117pp CI[0.05863548208440185, 0.8256043729525342]. All source-relative and spatial tradeoffs remain in the report.
Long events (>5s), 4243 queries: matched-MBR tIoU +5.922207pp CI[5.0116342136836245, 6.881713566093487]; vIoU +3.085558pp CI[2.5041530215105614, 3.6988014922867256]. All source-relative and spatial tradeoffs remain in the report.

The full official set is historically exposed through TA work. The Vid profile used 64 CG development queries; those 64 yield matched-MBR vIoU +3.610412pp CI[.719841,6.751060]. Remaining10,239 queries across732 parents yield +1.515969pp CI[1.152426,1.889056]. This is not a newly unseen benchmark or a parameter-free independent confirmation. CIs are pointwise parent-cluster bootstrap intervals without familywise correction. Historical source-calibration transfer and gamma-selection provenance limits remain unchanged.

## Verification and actual cost

All predictions sealed before independent CPU official labels. **51,515 official/dense checks**, maximum discrepancy **2.7755575615628914e-15**. Root actually ran the original audit.py report and the separate source-semantics adapter: **30,909 prediction/compact/receipt SHA checks**, every compact output array replayed exactly, and all reported means recomputed. Postseal reports perform zero new model, Source, expert or GT scoring. Both new native checkpoint/no-GT/head/cold-pool bridges passed. Raw disposal is restricted to this study's NEW completed scratch after exact replay; old raw roots are untouched. Compact sufficiency supports this frozen profile, not arbitrary future repooling.

Actual worker PID/start 2300285:238105449: 10303 queries, 320 legacy acquisitions, 9985 new Source forwards (includes two native reference extras), 9733 new joint same-expert8 forwards. Whole worker wall 46883.600194s; sum query pipeline 44797.827330s, Source capture 17129.610882s, feedback 5731.539542s. Expert original-cost provenance: {'expert_seconds': 1965.2564689512365, 'preprocess_seconds': 182.16034156270325, 'forward_seconds': 1779.3887878353707, 'postprocess_seconds': 3.7073395531624556}; teacher load 4.889353s. Source loads once per actual worker: {'2300285:238105449': 8.94419453991577}.

There are 9,983 newly acquired Source queries and 9,985 forwards including two references. New expert calls are9,733; compatible teacher-cache hits retain acquisition provenance and must not be interpreted as zero-cost deployment. Nested Source/teacher/feedback times are not additive. Shared-host/cache/research wall, including audit hooks, is not uncached deployment latency. All source/teacher costs and old acquisition provenance remain in the bound receipts.

See [main report](report.json), [all signed outcomes](all_signed.tsv), [severe/timegroups/rescues](all_signed_severe_timegroups.tsv), [audit receipt](reporting_receipt.json), [explicit original-Source comparisons](named_source_comparison/report.json), [selection queries](selection_queries/report.json) and [remaining queries](remaining_queries/report.json). Every positive and negative result is retained. Media, captions, coordinates, raw grids, masks and weights stay private.

The independent CG-Vid current-MAP consumer has a separate complete-domain scoring/replay audit. Its results cannot be inferred from this MBR study. Old strict-deletion queues are superseded and unexecuted; no automatic GPU follow-up is authorized.
