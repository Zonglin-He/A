# Temporal evidence quality and error risk: vidstg

All 10303 queries from the complete sealed/scored/cold-array-audited temporal controls. Offline GT event membership only, no action-label model or new acquisition. Independent audit rederives AUROC by explicit positive-negative pair comparisons, event masks and per-query means and CIs.

|Source tIoU / diagnostic group|Queries|Scorable corrected-ROI AUROC queries|Event evidence mean|Background evidence mean|Corrected-ROI AUROC [95% CI]|Corrected-minus-Source-ROI AUROC [95% CI]|Delta t pp [95% CI]|t harm % [95% CI]|
|---|---:|---:|---:|---:|---|---|---|---|
|all|10303|9511|0.11005785478400266|-0.09610129566559066|+0.641318 [+0.628854, +0.653472]|-0.000725 [-0.003153, +0.001744]|+3.189738 [+2.753594, +3.658697]|+28.710084 [+27.385588, +30.052741]|
|tIoU_0_02|4403|4228|-0.04465055117425052|0.013928372737227766|+0.442252 [+0.428101, +0.456240]|-0.000464 [-0.004280, +0.003390]|+3.765774 [+3.253128, +4.313531]|+17.874177 [+16.235017, +19.574694]|
|tIoU_02_05|3018|2637|0.17321278240969207|-0.09109679578465724|+0.715463 [+0.698760, +0.732486]|-0.000607 [-0.005131, +0.003757]|+4.801621 [+3.918697, +5.714526]|+33.499006 [+30.953850, +36.236351]|
|tIoU_05_08|2228|2025|0.27984948417161465|-0.2583698127107788|+0.871716 [+0.861268, +0.881832]|-0.001798 [-0.005678, +0.002266]|+2.155278 [+1.368459, +2.938207]|+36.086176 [+33.442065, +38.773861]|
|tIoU_08_10|654|621|0.2817456064418904|-0.3373389078662858|+0.930485 [+0.920093, +0.940018]|+0.000497 [-0.003678, +0.004930]|-4.602578 [-5.888238, -3.376195]|+54.434251 [+49.409923, +59.450739]|
|temporal_improved|4733|4118|0.18289740066995466|-0.16700771122398392|+0.753027 [+0.739015, +0.766234]|+0.005595 [+0.002259, +0.008828]|+11.846010 [+11.208582, +12.512363]|+0.000000 [+0.000000, +0.000000]|
|temporal_harmed|2958|2831|0.19083187075310815|-0.06937351304234567|+0.690710 [+0.673996, +0.707166]|-0.009853 [-0.013879, -0.005973]|-7.844252 [-8.248301, -7.448711]|+100.000000 [+100.000000, +100.000000]|

Source temporal-quality bins are fixed [0,.2), [.2,.5), [.5,.8), [.8,1]. Temporal gains/harm compare current output to native untouched Source; same-joint-readout and no-expert TR controls are separately available in the temporal-control table, avoiding decoder/readout attribution. AUROC is computed WITHIN each query, with tied scores contributing one half; only queries containing both GT event and background sampled frames are scorable. Missing/one-class queries remain in outcome denominators and their counts are explicit, never assigned AUROC zero or dropped from accuracy. Event/non-event means, frame support counts and all query scores are retained. Frames are not independent bootstrap units.

## Confidence and prototype separation, descriptive groups

|Group|Original joint-prior peak [95% CI]|Original joint-prior entropy [95% CI]|Source-ROI prototype separation [95% CI]|Corrected-ROI prototype separation [95% CI]|
|---|---|---|---|---|
|all|+0.014386 [+0.013654, +0.015158]|+5.970371 [+5.899822, +6.040691]|+0.616877 [+0.605493, +0.628670]|+0.614806 [+0.603566, +0.626394]|
|temporal_improved|+0.014353 [+0.013447, +0.015296]|+5.951586 [+5.869396, +6.034003]|+0.632478 [+0.618430, +0.646966]|+0.632620 [+0.618664, +0.646804]|
|temporal_harmed|+0.015625 [+0.014838, +0.016448]|+5.754822 [+5.681922, +5.828465]|+0.629046 [+0.613003, +0.645825]|+0.625779 [+0.610238, +0.642069]|

These groups condition on the already observed offline outcome and show associations only; q0 joint-prior peak is explicitly not an independently calibrated confidence probability. Prototype separation and appearance/event discrimination are within-video self-consistency measures because evaluated frames participate in prototype fitting. A positive AUROC, object-local contrast or temporal correspondence effect does NOT establish independent action-semantics understanding or causal mutual reinforcement. No online GT gate, default selection or algorithm changes are made.

A separate form x fixed-quality-thirds report separates temporal evidence at the SAME E trajectory (ExpertInitTR-FixedE_SourceTR), final spatial decoding (Ours-ExpertInitTR), and total gain (Ours-Source). Same E controlled time branches have identical frame coordinates; their vIoU difference reflects intervals, not a spatial-coordinate difference. All negative and null strata and the original interpolation comparison are retained. Pointwise parent10000 CIs, no familywise/unseen confirmation claim. Cached CPU cost is not deployment time; inherited Source/expert acquisition is nonzero.
