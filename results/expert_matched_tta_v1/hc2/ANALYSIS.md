# Identical eight expert observations, independently inferred full tubes: hc2

1902 queries / 227 parents; whole seal before 5706 new official metric checks (Source and Ours metrics inherited with exact array proofs), maximum discrepancy 9.9920072216264089e-16; independent 3804 prediction/receipt SHA checks plus all-parent/TTA provenance and all arrays cold rederived exact.

|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|---:|
|Source|40.141475|20.704678|50.685778|29.022082|8.937960|
|SourceExpertInterpolation|40.141475|21.886351|48.980489|31.177708|10.830705|
|TentExpertInterpolation|40.145299|21.886351|48.980489|31.177708|10.830705|
|SARExpertInterpolation|40.145299|21.886351|48.980489|31.177708|10.830705|
|Ours|49.764548|27.349044|53.283819|41.219769|16.876972|

|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe union|Rescue union|
|---|---:|---:|---:|---:|
|Ours-Source|+9.623073 [8.594747, 10.683117]|+6.644365 [5.902588, 7.408451]|458|1129|
|Ours-SourceExpertInterpolation|+9.623073 [8.594747, 10.683117]|+5.462692 [4.842616, 6.094744]|396|1061|
|Ours-TentExpertInterpolation|+9.619249 [8.591392, 10.679127]|+5.462692 [4.842616, 6.094744]|396|1061|
|Ours-SARExpertInterpolation|+9.619249 [8.591392, 10.679127]|+5.462692 [4.842616, 6.094744]|396|1061|
|TentExpertInterpolation-SourceExpertInterpolation|+0.003824 [0.000000, 0.012258]|+0.000000 [0.000000, 0.000000]|0|1|
|SARExpertInterpolation-SourceExpertInterpolation|+0.003824 [0.000000, 0.012258]|+0.000000 [0.000000, 0.000000]|0|1|
|SourceExpertInterpolation-Source|+0.000000 [0.000000, 0.000000]|+1.181673 [0.656306, 1.716009]|272|437|

Unlike the prior matched-time spatial table, each Source/Tent/SAR interpolation row uses its own original native per-offset MAP and physical-hull event interval. No Ours event interval or temporal-posterior correction is injected. All three share the exact original same-eight center/log-size interpolation tensor including the original E outside coverage; no DP is attached to interpolation. Ours retains its current structured spatial and temporal inference. The no-update SourceExpertInterpolation row separates expert fusion from optimizer effects. The current official pinned Tent/SAR STVG adapters update temporal-decoder LayerNorm affine only, with fixed Source boxes and five episodic steps; not canonical BN/full-network/continual method rankings. Actual parameter/logit changes and SAR empty reliable-filter cases are retained, never forced argmax changes.

All t/v/s/R.3/R.5 signed metrics, parent-cluster 10000 bootstrap pointwise confidence intervals (not familywise), severe >5pp t/v union once/rescues, short <=5s and long >5s are preserved. Full datasets are historically exposed, no parameter or sample sweep. These results address same-observation evidence usage, not equal compute budgets.

## Duration-specific outcomes

### short_le5s, 544 queries

- Ours-Source: t +3.539258 [1.682666, 5.388260], v +3.434542 [2.062777, 4.774484] pp.
- Ours-SourceExpertInterpolation: t +3.539258 [1.682666, 5.388260], v +2.139138 [1.002190, 3.302905] pp.
- Ours-TentExpertInterpolation: t +3.539258 [1.682666, 5.388260], v +2.139138 [1.002190, 3.302905] pp.
- Ours-SARExpertInterpolation: t +3.539258 [1.682666, 5.388260], v +2.139138 [1.002190, 3.302905] pp.
- TentExpertInterpolation-SourceExpertInterpolation: t +0.000000 [0.000000, 0.000000], v +0.000000 [0.000000, 0.000000] pp.
- SARExpertInterpolation-SourceExpertInterpolation: t +0.000000 [0.000000, 0.000000], v +0.000000 [0.000000, 0.000000] pp.
- SourceExpertInterpolation-Source: t +0.000000 [0.000000, 0.000000], v +1.295404 [0.358437, 2.211335] pp.
### long_gt5s, 1358 queries

- Ours-Source: t +12.060182 [10.872964, 13.264188], v +7.930185 [7.010995, 8.893365] pp.
- Ours-SourceExpertInterpolation: t +12.060182 [10.872964, 13.264188], v +6.794072 [6.071963, 7.520692] pp.
- Ours-TentExpertInterpolation: t +12.054827 [10.867026, 13.259435], v +6.794072 [6.071963, 7.520692] pp.
- Ours-SARExpertInterpolation: t +12.054827 [10.867026, 13.259435], v +6.794072 [6.071963, 7.520692] pp.
- TentExpertInterpolation-SourceExpertInterpolation: t +0.005355 [0.000000, 0.017427], v +0.000000 [0.000000, 0.000000] pp.
- SARExpertInterpolation-SourceExpertInterpolation: t +0.005355 [0.000000, 0.017427], v +0.000000 [0.000000, 0.000000] pp.
- SourceExpertInterpolation-Source: t +0.000000 [0.000000, 0.000000], v +1.136113 [0.534679, 1.745419] pp.

## Measured and missing costs

{
  "prediction_worker": {
    "pid": 390225,
    "startticks": 255906546
  },
  "cached_composition_elapsed_wall_seconds": 88.23565775481984,
  "official_score_worker": {
    "pid": 392268,
    "startticks": 255915597
  },
  "official_score_elapsed_wall_seconds": 38.89899026369676,
  "independent_audit_worker": {
    "pid": 393337,
    "startticks": 255920136
  },
  "audit_elapsed_wall_seconds": 40.374738027341664,
  "original_TTA_costs": {
    "2558667:246462743": {
      "queries": 1902,
      "reused_completed": 128,
      "new_Source": 1608,
      "shared_Source": 0,
      "new_optimizer_iterations": 17740,
      "adaptation_seconds": 444.79006355721503
    }
  },
  "new_Source": 0,
  "new_expert": 0,
  "new_optimizer": 0,
  "original_expert_and_Source_acquisition_nonzero": true,
  "uncached_complete_combination_latency_seconds": null,
  "latency_status": "Not yet measured for new combinations. Saved composition/score/audit elapsed wall is not deployment; inherited nested acquisition/optimizer/Source/teacher costs must not be added twice. Ours independently measured latency is not a substitute for these combinations."
}
