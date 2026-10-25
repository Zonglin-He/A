# Current no-shrinkage CG: hc2

Complete whole-domain seal → independent official scoring → exact array rederivation and all-mean audit: **1902 queries / 227 parents**, 5706 official dense checks (max discrepancy 1.4432899320127035e-15), 3804 prediction/receipt SHA checks plus bound parent hashes verified during derivation. Zero new Source, expert, optimizer or GT scoring in this root export.

Kappa0 is the directly user-selected current method. Selection followed exposed TA full results; no claim of independent unseen confirmation. Old kappa64 scientific files and outputs remain immutable. Source uses native per-offset MAP and physical hull; SourceJointMAP isolates readout. Every changed posterior uses its own MAP interval and actually re-solves spatial DP.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|
|---|---:|---:|---:|---:|---|
|Source|23.220480|14.673932|18.401682|7.465825|0/0/0|
|SourceJointMAP|23.220480|14.673932|18.401682|7.465825|0/0/0|
|FullMAP|43.938130|26.977349|39.379600|17.718191|147/119/164|

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|FullMAP-Source|+20.717649 [19.179888, 22.311567]|+12.303417 [11.236374, 13.433269]|147/119/164|1126|
|FullMAP-SourceJointMAP|+20.717649 [19.179888, 22.311567]|+12.303417 [11.236374, 13.433269]|147/119/164|1126|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / FullMAP-Source|544|+11.712495 [9.618708, 13.840843]|+7.015239 [5.607713, 8.390129]|
|short_le5s / FullMAP-SourceJointMAP|544|+11.712495 [9.618708, 13.840843]|+7.015239 [5.607713, 8.390129]|
|long_gt5s / FullMAP-Source|1358|+24.325016 [22.488870, 26.149691]|+14.421803 [13.135411, 15.714147]|
|long_gt5s / FullMAP-SourceJointMAP|1358|+24.325016 [22.488870, 26.149691]|+14.421803 [13.135411, 15.714147]|

## Fixed controls and interpretation

CG uses its existing per-domain profile with **only kappa set to0**; the full profile is in fixed_domain_profile.json. HC gamma0/u0/link0 deliberately disables unobserved propagation and linkage, and every unobserved box remains the original Source box. Its unobserved fixed-support sIoU change is exactly zero, so this is not evidence that complete propagation works on HC. All227 HC parents were used in development; remaining queries do not create independent parent validation. Source weight/input/local precision qualifications are inherited from the original study.

Relative to historical kappa64 jointMAP+DP, delta t +5.685486 [4.875689, 6.515200] pp and delta v +3.494738 [2.962349, 4.043958] pp. The mean change and all severe/rescue/short-long outcomes are in legacy_kappa_comparison.json and the anonymous signed TSV; neither historical negative outcomes nor tail damage are removed.

## Costs and evidence

{
  "new_Source": 0,
  "new_expert": 0,
  "new_optimizer": 0,
  "current_worker_CPU_seconds": {
    "1709802:245670587": 45.439835514407605
  },
  "scope": "Conditional cached-query CPU wall excludes setup/outer-loop, may include shared-host effects. Original Source/expert acquisitions remain nonzero; nested costs must not be added or used as current deployment latency.",
  "inherited_acquisition_audit": {
    "path": "outputs/cgstvg_full_v1/hc2_reporting_receipt.json",
    "sha256": "fe28a63394fbc717d873bf51004c7c733d1b758b155f8bfa9a9c9ab7f32f5d92",
    "original_worker_segments": {
      "1427705:237101753": {
        "queries": 1902,
        "reused_legacy": 227,
        "feedback_seconds": 1057.9093749206513,
        "pipeline_seconds": 9726.43451165082,
        "new_Source_forwards": 1677,
        "new_Source_seconds": 3658.7994819986634,
        "new_expert": 1675,
        "expert_original_cost": {
          "expert_seconds": 359.12877250509337,
          "preprocess_seconds": 40.75799171673134,
          "forward_seconds": 317.53540577786043,
          "postprocess_seconds": 0.8353750105015934
        },
        "teacher_load_seconds": 5.528525684028864
      }
    }
  }
}

Confidence intervals are pointwise parent-cluster bootstrap intervals, not familywise. Short means native-GT duration <=5s. Severe means a signed tIoU or vIoU loss >5pp, union counted once; rescues use a >5pp gain. All signed positives and negatives, all means, CIs and tails remain in the public report/TSVs. Private captions/media/coordinates/features/weights are excluded.
