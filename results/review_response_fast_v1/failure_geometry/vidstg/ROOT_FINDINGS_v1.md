# vidstg: exact full-target output geometry census

10303 complete previously audited query outputs; no new model, ground-truth file or official score. Values are geometric failures/proxies, not tracked identity-switch, occlusion or expert-correspondence labels. A box crossing an image edge is distinct from a positive-area box wholly outside the image; both denominators are given. Original out-of-image predictions are never silently clipped.

|Arm|Support|Frames|Queries with partial outside|Fully outside frames|Queries with fully outside|Empty frames|
|---|---|---:|---:|---:|---:|---:|
|Source|all_native|1138800|1584|0|0|0|
|Source|own_event|368857|766|0|0|0|
|FullMAP|all_native|1138800|7309|33677|1029|0|
|FullMAP|own_event|497751|6074|6253|619|0|
|FinalIndependent|all_native|1138800|7359|33669|1029|0|
|FinalIndependent|own_event|497751|6183|6245|618|0|

All-query anonymized counts include observed/unobserved support and all native versus each arm's own predicted event. Source/Full/Independent time intervals can differ; comparisons of own-event fractions therefore are descriptive, not a fixed-support causal attribution. To claim true identity-switch/occlusion prevalence requires independent suitable labels; no such rates are fabricated from low IoU or trajectory jumps. Existing complete signed severe-loss/rescue and actual negative images are retained. No fallback or threshold was selected on scores.
