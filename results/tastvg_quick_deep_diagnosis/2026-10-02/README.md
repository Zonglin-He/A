# Posthoc CPU GT pipeline analysis of the sealed best-config quick panel

VidSTG: 670 sources / one query each / two orders / six conditions / 8040 arrivals. HC-STVG-v2: 128 sources / one query each / one order / six conditions / 768 arrivals. Same-domain official TA-STVG checkpoints, sparse independent temporal/spatial experts at 25%, previously sealed per-dataset best configurations. All sources historically exposed. This directory contains analysis, not a new model experiment or an official all-query evaluation.

All 8808 predictions were globally sealed before GT scoring. Base results and full anonymous score/step logs are at `results/tastvg_best_quick/2026-10-01` in commit `002bab8d9e7dd178649fddb00a311bcbd3c41122`. The new CPU analysis verified the 2016/192 scheduled sealed payloads and matching expert caches. Model forward calls, GPU initialization, parameter updates and new predictions: zero. No parameter reselection, online GT gate, method promotion, or resume of paused all-query work.

Anonymous DEEP_STEPS / REFERENCE_ROWS / TEMPORAL_ROWS preserve all descriptive scalar observations; CASES include large positive and negative examples selected posthoc. Quantile arrays are min /25% /median /75% /max. In SUMMARY, flat net_GT_delta_pp is 100 times the SUM of step deltas, not a cohort mean. Its proper mean appears in SUPPLEMENTARY. Unique teacher winner better but harmful updates are 159/11 on corruption first steps; the broader Vid top-set count179 contains20 flat cases and must not be described as159+20 informative preferences.

Reference GT-IoU is measured only on sampled valid reference frames with scored GT boxes, not whole-tube teacher quality. Empty/intersection-zero support, threshold-based cases and reward-band groups are diagnostic descriptions. Repeated sources, orders and candidate pairs are correlated. Current-update post predictions are diagnostic and are not online outputs. Proposed mechanisms are hypotheses, not demonstrated improvements.

Read `docs/TA_QUICK_DEEP_GT_ANALYSIS.md` for findings, primary literature and explicit transfer limitations. `docs/TA_FULL_CACHE_REUSE.md` corrects the actual teacher feature extractor to PE-Core-L14-336; its first version incorrectly called it CLIP. No teacher or result changed.

Reconstruct and verify anonymous scalars (requires NumPy, no GPU, GT or raw model cache):

    python -B scripts/summarize_tastvg_quick_deep_cpu_v1.py results/tastvg_quick_deep_diagnosis/2026-10-02/vidstg results/tastvg_best_quick/2026-10-01/vidstg
    python -B scripts/summarize_tastvg_quick_deep_cpu_v1.py results/tastvg_quick_deep_diagnosis/2026-10-02/hc2 results/tastvg_best_quick/2026-10-01/hc2

The private full GT log analysis script requires authorized local media/annotations and original sealed payloads; public scalar reconstruction does not. Restricted inputs, real source/caption identities, GT coordinates, raw predictions/state tensors and weights are excluded.
