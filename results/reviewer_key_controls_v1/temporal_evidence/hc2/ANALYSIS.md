# Temporal evidence quality and error risk: hc2

All 1902 queries from the complete sealed/scored/cold-array-audited temporal controls. Offline GT event membership only, no action-label model or new acquisition. Independent audit rederives AUROC by explicit positive-negative pair comparisons, event masks and per-query means and CIs.

|Source tIoU / diagnostic group|Queries|Scorable corrected-ROI AUROC queries|Event evidence mean|Background evidence mean|Corrected-ROI AUROC [95% CI]|Corrected-minus-Source-ROI AUROC [95% CI]|Delta t pp [95% CI]|t harm % [95% CI]|
|---|---:|---:|---:|---:|---|---|---|---|
|all|1902|1901|0.23842092940172477|-0.15966934272840644|+0.740260 [+0.728046, +0.752783]|+0.000335 [-0.001205, +0.001831]|+9.623073 [+8.594747, +10.683117]|+26.182965 [+24.010538, +28.384069]|
|tIoU_0_02|474|474|-0.08103736803970132|0.01046288101446321|+0.434042 [+0.409514, +0.458157]|+0.001226 [-0.002575, +0.004759]|+12.368883 [+10.861795, +13.877454]|+6.540084 [+4.149355, +8.961442]|
|tIoU_02_05|710|710|0.2726679553287151|-0.17193157651056676|+0.782962 [+0.766175, +0.799931]|+0.000111 [-0.002132, +0.002383]|+16.992322 [+15.533362, +18.503855]|+19.718310 [+16.931707, +22.628726]|
|tIoU_05_08|581|580|0.4086297165067239|-0.2534309820579031|+0.890967 [+0.877559, +0.903206]|-0.000650 [-0.002794, +0.001426]|+3.195017 [+1.452151, +4.816869]|+40.619621 [+36.274427, +45.302013]|
|tIoU_08_10|137|137|0.44438107006645766|-0.28780611103262815|+0.940398 [+0.927429, +0.952085]|+0.002579 [-0.001206, +0.007608]|-10.807447 [-13.394244, -8.196619]|+66.423358 [+58.331954, +74.285714]|
|temporal_improved|1195|1194|0.3017831010842875|-0.2190586092724074|+0.811536 [+0.799384, +0.823664]|+0.000469 [-0.001166, +0.002105]|+20.565384 [+19.578122, +21.605102]|+0.000000 [+0.000000, +0.000000]|
|temporal_harmed|498|498|0.26588794054242293|-0.09657305640987973|+0.729746 [+0.708258, +0.752383]|+0.000615 [-0.002563, +0.003824]|-12.595483 [-13.580735, -11.603949]|+100.000000 [+100.000000, +100.000000]|

Source temporal-quality bins are fixed [0,.2), [.2,.5), [.5,.8), [.8,1]. Temporal gains/harm compare current output to native untouched Source; same-joint-readout and no-expert TR controls are separately available in the temporal-control table, avoiding decoder/readout attribution. AUROC is computed WITHIN each query, with tied scores contributing one half; only queries containing both GT event and background sampled frames are scorable. Missing/one-class queries remain in outcome denominators and their counts are explicit, never assigned AUROC zero or dropped from accuracy. Event/non-event means, frame support counts and all query scores are retained. Frames are not independent bootstrap units.

## Confidence and prototype separation, descriptive groups

|Group|Original joint-prior peak [95% CI]|Original joint-prior entropy [95% CI]|Source-ROI prototype separation [95% CI]|Corrected-ROI prototype separation [95% CI]|
|---|---|---|---|---|
|all|+0.015667 [+0.015017, +0.016334]|+5.730482 [+5.698912, +5.761491]|+0.693401 [+0.680083, +0.707094]|+0.694546 [+0.681191, +0.708135]|
|temporal_improved|+0.015875 [+0.015122, +0.016676]|+5.716729 [+5.679175, +5.752555]|+0.708916 [+0.693859, +0.724703]|+0.710498 [+0.695507, +0.726177]|
|temporal_harmed|+0.012464 [+0.011755, +0.013257]|+5.839471 [+5.797272, +5.879719]|+0.670427 [+0.649293, +0.692395]|+0.670510 [+0.649008, +0.692857]|

These groups condition on the already observed offline outcome and show associations only; q0 joint-prior peak is explicitly not an independently calibrated confidence probability. Prototype separation and appearance/event discrimination are within-video self-consistency measures because evaluated frames participate in prototype fitting. A positive AUROC, object-local contrast or temporal correspondence effect does NOT establish independent action-semantics understanding or causal mutual reinforcement. No online GT gate, default selection or algorithm changes are made.

A separate form x fixed-quality-thirds report separates temporal evidence at the SAME E trajectory (ExpertInitTR-FixedE_SourceTR), final spatial decoding (Ours-ExpertInitTR), and total gain (Ours-Source). Same E controlled time branches have identical frame coordinates; their vIoU difference reflects intervals, not a spatial-coordinate difference. All negative and null strata and the original interpolation comparison are retained. Pointwise parent10000 CIs, no familywise/unseen confirmation claim. Cached CPU cost is not deployment time; inherited Source/expert acquisition is nonzero.
