# Current no-shrinkage CG: vidstg

Complete whole-domain seal → independent official scoring → exact array rederivation and all-mean audit: **10303 queries / 732 parents**, 30909 official dense checks (max discrepancy 2.6645352591003757e-15), 20606 prediction/receipt SHA checks plus bound parent hashes verified during derivation. Zero new Source, expert, optimizer or GT scoring in this root export.

Kappa0 is the directly user-selected current method. Selection followed exposed TA full results; no claim of independent unseen confirmation. Old kappa64 scientific files and outputs remain immutable. Source uses native per-offset MAP and physical hull; SourceJointMAP isolates readout. Every changed posterior uses its own MAP interval and actually re-solves spatial DP.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|
|---|---:|---:|---:|---:|---|
|Source|17.318935|7.532313|7.017374|1.912064|0/0/0|
|SourceJointMAP|17.318935|7.532313|7.017374|1.912064|0/0/0|
|FullMAP|28.490948|12.056521|15.286810|5.949723|1020/1034/1486|

|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|
|---|---:|---:|---|---:|
|FullMAP-Source|+11.172013 [10.262910, 12.126636]|+4.524208 [4.001660, 5.082509]|1020/1034/1486|4908|
|FullMAP-SourceJointMAP|+11.172013 [10.262910, 12.126636]|+4.524208 [4.001660, 5.082509]|1020/1034/1486|4908|

## Duration groups

|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---:|---:|---:|
|short_le5s / FullMAP-Source|6060|+5.563906 [4.647648, 6.548462]|+1.714081 [1.198050, 2.263848]|
|short_le5s / FullMAP-SourceJointMAP|6060|+5.563906 [4.647648, 6.548462]|+1.714081 [1.198050, 2.263848]|
|long_gt5s / FullMAP-Source|4243|+19.181707 [17.798484, 20.554038]|+8.537728 [7.658298, 9.447125]|
|long_gt5s / FullMAP-SourceJointMAP|4243|+19.181707 [17.798484, 20.554038]|+8.537728 [7.658298, 9.447125]|

## Fixed controls and interpretation

CG uses its existing per-domain profile with **only kappa set to0**; the full profile is in fixed_domain_profile.json. HC keeps gamma0/u0/link0/beta.5/context1.5, so its unobserved boxes remain Source and fixed-support unobserved change is exactly0; it cannot establish full propagation effectiveness. Vid keeps its distinct gamma.5/u.5/link.25/beta1/context1 profile, with propagation enabled. All227 HC parents were developed; Vid used64developmentqueries and historical full exposure. The per-domain profile file is authoritative; remaining queries do not create independent-unseen validation. Source weight/input/local precision qualifications are inherited from the original study.

Relative to historical kappa64 jointMAP+DP, delta t +3.318887 [2.901512, 3.752279] pp and delta v +1.263592 [1.060460, 1.476110] pp. The mean change and all severe/rescue/short-long outcomes are in legacy_kappa_comparison.json and the anonymous signed TSV; neither historical negative outcomes nor tail damage are removed.

## Costs and evidence

{
  "new_Source": 0,
  "new_expert": 0,
  "new_optimizer": 0,
  "current_worker_CPU_seconds": {
    "1740823:245701590": 246.10766760865226
  },
  "scope": "Conditional cached-query CPU wall excludes setup/outer-loop, may include shared-host effects. Original Source/expert acquisitions remain nonzero; nested costs must not be added or used as current deployment latency.",
  "inherited_acquisition_audit": {
    "path": "outputs/cgstvg_full_v1/vidstg_reporting_receipt.json",
    "sha256": "3e10cc4124b501635c49cf33868aea891d6698e6d55d4f7bb42f055238ba4e43",
    "original_worker_segments": {
      "2300285:238105449": {
        "queries": 10303,
        "reused_legacy": 320,
        "feedback_seconds": 5731.5395415653475,
        "pipeline_seconds": 44797.82732954528,
        "new_Source_forwards": 9985,
        "new_Source_seconds": 17129.61088172812,
        "new_expert": 9733,
        "expert_original_cost": {
          "expert_seconds": 1965.2564689512365,
          "preprocess_seconds": 182.16034156270325,
          "forward_seconds": 1779.3887878353707,
          "postprocess_seconds": 3.7073395531624556
        },
        "teacher_load_seconds": 4.889352501835674
      }
    }
  }
}

Confidence intervals are pointwise parent-cluster bootstrap intervals, not familywise. Short means native-GT duration <=5s. Severe means a signed tIoU or vIoU loss >5pp, union counted once; rescues use a >5pp gain. All signed positives and negatives, all means, CIs and tails remain in the public report/TSVs. Private captions/media/coordinates/features/weights are excluded.
