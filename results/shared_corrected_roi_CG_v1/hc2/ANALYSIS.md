# CG shared-corrected-box ROI transfer: hc2

Whole 1902 queries / 227 parents sealed before independent official labels. 7608 official checks, maximum error 1.4432899320127035e-15; 5706 prediction/compact/receipt SHA checks plus bound old inputs/teacher provenance, cold all-array replay and all means verified. Root export reads audited scalars and costs only, no model/Source/expert/optimizer/GT scoring.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs corrupted-or-native Source|
|---|---:|---:|---:|---:|---|
|Source|23.220480|14.673932|18.401682|7.465825|0/0/0|
|SourceJointMAP|23.220480|14.673932|18.401682|7.465825|0/0/0|
|HistoricalTwoDP|43.938130|26.977349|39.379600|17.718191|147/119/164|
|FullMAP|43.995003|26.974277|39.432177|17.770768|150/117/167|

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|FullMAP-Source|+20.774523 [19.257968, 22.354213]|+12.300345 [11.252079, 13.408554]|150/117/167|1129|
|FullMAP-SourceJointMAP|+20.774523 [19.257968, 22.354213]|+12.300345 [11.252079, 13.408554]|150/117/167|1129|
|FullMAP-HistoricalTwoDP|+0.056874 [-0.148949, 0.276751]|-0.003072 [-0.103155, 0.087301]|25/10/25|23|

## Duration groups

Short is GT duration <=5 seconds, long is >5 seconds. All intervals are pointwise parent-cluster bootstrap95%, not familywise.

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / FullMAP-Source|544|+11.788544 [9.706523, 13.886968]|+7.015146 [5.617522, 8.384475]|
|short_le5s / FullMAP-SourceJointMAP|544|+11.788544 [9.706523, 13.886968]|+7.015146 [5.617522, 8.384475]|
|short_le5s / FullMAP-HistoricalTwoDP|544|+0.076049 [-0.085735, 0.276041]|-0.000093 [-0.071229, 0.074994]|
|long_gt5s / FullMAP-Source|1358|+24.374208 [22.578475, 26.165802]|+14.417538 [13.164861, 15.681004]|
|long_gt5s / FullMAP-SourceJointMAP|1358|+24.374208 [22.578475, 26.165802]|+14.417538 [13.164861, 15.681004]|
|long_gt5s / FullMAP-HistoricalTwoDP|1358|+0.049192 [-0.220869, 0.346190]|-0.004265 [-0.139136, 0.119259]|

Every signed positive/negative result, fixed-support observed/unobserved metric and boundary error remains in report.json and the signed TSVs. Severe means a loss greater than5pp; union counts a query once. Rescue means an increase greater than5pp in t or v. Source is native per-offset legal MAP plus physical hull. Historically exposed cohorts and user-directed method selection are not unseen confirmation; zero-crossing intervals are not equivalence proofs.

## Method and controls

HC gamma0/u0/link0/beta.5/context1.5; unobserved box coordinates remain Source, so no full spatial-propagation efficacy claim. Kappa0 remains fixed. Direct actual corrected-box ROI pooling replaces the first auxiliary DP; final interval-conditioned trajectory DP remains. HistoricalTwoDP is the immutable earlier kappa0 method with exactly the same domain profile; SourceJointMAP separates readout from adaptation. Hidden original-array/pool compatibility is a prerequisite, not a new MBR experiment or selected auxiliary-DP cost. No new parameter search or independent-unseen validation.

## Costs

{
  "actual_counts": {
    "queries": 1902,
    "reused_raw": 227,
    "new_Source": 1675,
    "new_Source_forwards": 1675,
    "new_expert": 0
  },
  "actual_worker": {
    "pid": 3429690,
    "startticks": 253613860
  },
  "whole_worker_elapsed_seconds": 7790.221312953159,
  "model_load_once_for_actual_worker": {
    "target": "hc2",
    "source": "vidstg",
    "checkpoint": "vidstg.pth",
    "sha256": "cbf3ee4036dab88de3314dc3a520ab0eb2b4d292a2796d12d0f3493c39eed9ef",
    "model_key": "model_ema",
    "all_model_state_loaded": true,
    "ignored_deterministic_extras": [
      "text_encoder.body.embeddings.position_ids"
    ],
    "precision": "original_FP32_no_autocast",
    "resolution": 420,
    "single_native_grid": true,
    "source_config_yaml": "vidstg.yaml",
    "original_source_thresholds_preserved": true,
    "actual_load_seconds": 8.596691225189716,
    "worker": {
      "pid": 3429690,
      "startticks": 253613860
    }
  },
  "actual_PID_start_segments": {
    "3429690:253613860": {
      "queries": 1902,
      "legacy_raw_reused": 227,
      "new_Source": 1675,
      "new_expert": 0,
      "Source_seconds": 3347.6331557920203,
      "pipeline_seconds": 7501.290022974368,
      "compatibility_seconds": 992.0044054468162,
      "selected_finish_seconds": 43.25216154521331
    }
  },
  "original_Source_teacher_acquisition_nonzero": true,
  "selected_first_auxiliary_DP": false,
  "scope": "Source/pipeline/compatibility/selected-finish/load clocks are nested as recorded and must not be summed. Reused raw captures retain acquisition costs. These conditional research walls are not uncached deployment latency. Failed model/paid attempts remain separate evidence; no blind retries. Sufficient states cannot support arbitrary future repooling after only owned new raw disposal."
}
