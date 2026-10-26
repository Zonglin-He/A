# Identical eight expert observations, independently inferred full tubes: vidstg

10303 queries / 732 parents; whole seal before 30909 new official metric checks (Source and Ours metrics inherited with exact array proofs), maximum discrepancy 3.219646771412954e-15; independent 20606 prediction/receipt SHA checks plus all-parent/TTA provenance and all arrays cold rederived exact.

|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|---:|
|Source|31.728454|11.138484|32.729055|12.831214|3.348539|
|SourceExpertInterpolation|31.728454|13.507401|36.072277|17.082403|7.240610|
|TentExpertInterpolation|31.728070|13.507256|36.072277|17.082403|7.240610|
|SARExpertInterpolation|31.728357|13.507371|36.072277|17.082403|7.240610|
|Ours|34.918192|14.558151|36.090153|19.411822|8.463554|

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe union|Rescue union|
|---|---:|---:|---:|---:|
|Ours-Source|+3.189738 [2.753594, 3.658697]|+3.419668 [3.009582, 3.851146]|2294|4460|
|Ours-SourceExpertInterpolation|+3.189738 [2.753594, 3.658697]|+1.050750 [0.802237, 1.310876]|1963|3489|
|Ours-TentExpertInterpolation|+3.190122 [2.753811, 3.659206]|+1.050896 [0.802245, 1.311042]|1963|3489|
|Ours-SARExpertInterpolation|+3.189836 [2.753779, 3.658699]|+1.050780 [0.802241, 1.310877]|1963|3489|
|TentExpertInterpolation-SourceExpertInterpolation|-0.000384 [-0.001762, 0.000599]|-0.000145 [-0.000567, 0.000120]|1|0|
|SARExpertInterpolation-SourceExpertInterpolation|-0.000098 [-0.000303, 0.000000]|-0.000030 [-0.000093, 0.000000]|0|0|
|SourceExpertInterpolation-Source|+0.000000 [0.000000, 0.000000]|+2.368917 [2.008753, 2.746894]|1206|2587|

Unlike the prior matched-time spatial table, each Source/Tent/SAR interpolation row uses its own original native per-offset MAP and physical-hull event interval. No Ours event interval or temporal-posterior correction is injected. All three share the exact original same-eight center/log-size interpolation tensor including the original E outside coverage; no DP is attached to interpolation. Ours retains its current structured spatial and temporal inference. The no-update SourceExpertInterpolation row separates expert fusion from optimizer effects. The current official pinned Tent/SAR STVG adapters update temporal-decoder LayerNorm affine only, with fixed Source boxes and five episodic steps; not canonical BN/full-network/continual method rankings. Actual parameter/logit changes and SAR empty reliable-filter cases are retained, never forced argmax changes.

All t/v/s/R.3/R.5 signed metrics, parent-cluster 10000 bootstrap pointwise confidence intervals (not familywise), severe >5pp t/v union once/rescues, short <=5s and long >5s are preserved. Full datasets are historically exposed, no parameter or sample sweep. These results address same-observation evidence usage, not equal compute budgets.

## Duration-specific outcomes

### short_le5s, 6060 queries

- Ours-Source: t -0.129647 [-0.447674, 0.187245], v +1.135805 [0.785926, 1.511448] pp.
- Ours-SourceExpertInterpolation: t -0.129647 [-0.447674, 0.187245], v -0.333687 [-0.518701, -0.147782] pp.
- Ours-TentExpertInterpolation: t -0.129918 [-0.448031, 0.187217], v -0.333734 [-0.518846, -0.147823] pp.
- Ours-SARExpertInterpolation: t -0.129480 [-0.447426, 0.187372], v -0.333636 [-0.518652, -0.147713] pp.
- TentExpertInterpolation-SourceExpertInterpolation: t +0.000272 [-0.000489, 0.001356], v +0.000047 [-0.000150, 0.000299] pp.
- SARExpertInterpolation-SourceExpertInterpolation: t -0.000166 [-0.000520, 0.000000], v -0.000051 [-0.000159, 0.000000] pp.
- SourceExpertInterpolation-Source: t +0.000000 [0.000000, 0.000000], v +1.469492 [1.098146, 1.862659] pp.
### long_gt5s, 4243 queries

- Ours-Source: t +7.930598 [7.102139, 8.789204], v +6.681559 [5.972095, 7.420513] pp.
- Ours-SourceExpertInterpolation: t +7.930598 [7.102139, 8.789204], v +3.028052 [2.511164, 3.562308] pp.
- Ours-TentExpertInterpolation: t +7.931918 [7.102557, 8.791325], v +3.028471 [2.511606, 3.563266] pp.
- Ours-SARExpertInterpolation: t +7.930598 [7.102139, 8.789204], v +3.028052 [2.511164, 3.562308] pp.
- TentExpertInterpolation-SourceExpertInterpolation: t -0.001320 [-0.004146, 0.000000], v -0.000420 [-0.001318, 0.000000] pp.
- SARExpertInterpolation-SourceExpertInterpolation: t +0.000000 [0.000000, 0.000000], v +0.000000 [0.000000, 0.000000] pp.
- SourceExpertInterpolation-Source: t +0.000000 [0.000000, 0.000000], v +3.653507 [3.094284, 4.239896] pp.

## Measured and missing costs

{
  "prediction_worker": {
    "pid": 394392,
    "startticks": 255924677
  },
  "cached_composition_elapsed_wall_seconds": 3921.0977674941532,
  "official_score_worker": {
    "pid": 705946,
    "startticks": 256317657
  },
  "official_score_elapsed_wall_seconds": 315.2870251610875,
  "independent_audit_worker": {
    "pid": 733409,
    "startticks": 256350519
  },
  "audit_elapsed_wall_seconds": 1608.20986467693,
  "original_TTA_costs": {
    "3432864:247190991": {
      "queries": 10303,
      "reused_completed": 128,
      "new_Source": 0,
      "shared_Source": 9988,
      "new_optimizer_iterations": 101750,
      "adaptation_seconds": 3357.7014066115953
    }
  },
  "new_Source": 0,
  "new_expert": 0,
  "new_optimizer": 0,
  "original_expert_and_Source_acquisition_nonzero": true,
  "uncached_complete_combination_latency_seconds": null,
  "latency_status": "Not yet measured for new combinations. Saved composition/score/audit elapsed wall is not deployment; inherited nested acquisition/optimizer/Source/teacher costs must not be added twice. Ours independently measured latency is not a substitute for these combinations."
}
