# Round2 fixed18 development: structured surface adapter v1

Status: completed, all144arm arrays sealed before separate-process numericGT scoring. Parent independent audit pending. Confirmation36 has not been inferred or scored. No method is adopted.

| Arm | mIoU% | Acc>.25 | Acc>.50 |
|---|---:|---:|---:|
| Source |30.9823|16/18|1/18|
| Fixed PCA |38.7691|12/18|7/18|
| Fixed surface selection |38.4834|12/18|7/18|
| Direct partial-face latent |30.9816|16/18|1/18|
| Adapter0 |30.9823|16/18|1/18|
| Adapter1 |26.2361|14/18|0/18|
| Adapter3 |21.8654|13/18|0/18|
| Source-target control3 |30.9823|16/18|1/18|

Geometry passes for10candidates in3/6Waymo frames,0/6Drone and0/6Quad. Only3frames have nonzero parameter updates; all3 lose IoU. No directlatent or adapter identity switch occurs. The two-vertical-face rule therefore provides little coverage, particularly for top-down observations. It has not solved same-class instance identity. PCA/surface improve strict-threshold counts but also lose4loose-threshold successes; averaging alone hides that tradeoff.

The worst parameter-update tail is sourceIoU.786879→adapter1.249691→adapter3.0. Another is.454691→.0. The directteacher remains near source, so these are not evidence that the geometry teacher intrinsically requires such large errors. The shared-adapter replay is bitexact to alloriginal0/1/3outputs. It shows unaccepted candidates drifting up to.79m and~1.57rad;2of3changed source-chosenqueries are themselves unaccepted by teacher. A teacher rejection gate did not protect the network output.

`DEV_OPTIMIZATION_DIAGNOSIS.json` identifies an optimizer-conditioning explanation. RMS-normalized288-dimensional feature vectors haveL1norm220–235. An Adam coordinate step of.01 can therefore change a query's scalar residual by order2, driving tanh close to saturation. The actual label-free loss increases immediately on all3updated samples: .0286→1.4017, .2649→1.7611, and .2251→.4944. Later losses remain above initial. Nonzero gradients/parameter changes alone had not established that the specified adaptation problem was even being optimized. This is an observed implementation/optimizer failure, distinct from limited teacher coverage and from route-level impossibility.

The only proposed bounded correction is function-space trust-region/backtracking for the existing adapter objective: keep teacher, gates, features, cohorts, source/PCA/surface/directlatent, steps and nominal Adam LR unchanged; scale each proposed parameter step using its maximum affine-output change and reject/backtrack unless the same unlabeled objective decreases. A predeclared maximum normalized residual change can bound yaw/center movement. This is an optimization repair, not a claim of newsemantic evidence or a newTTA contribution. No correction has yet been executed; no threshold/LR grid is proposed. Its exact constant/line-search rule and newcode must be locked before a single devrerun; retain v1 and the same confirmation36.

Engineering history:18predictions were computed before a final traceJSON `np.bool_`serialization failure. The failed unsealed directory and log remain. A newscalar-serializer wrapper left allscientificcode unchanged; all18×8rerun arrays were bitexact before repairedseal. Source repeat arrays and adapter0/source-targetcontrol3 are bitexact. This serialization fix changed no task predictions or evaluation rules.

Evidence: `dev_source/sealed.json`, `dev_predictions/sealed.json`, `dev_scored.json`, `dev_shared_adapter_diagnostic.json`, `DEV_OPTIMIZATION_DIAGNOSIS.json`, `SERIALIZATION_REPAIR_VERIFIED.json`. Scores are based on18frames in8sceneclusters; Drone/Quad only1development scene each, and alltheirvalidation scenes were exposed inpreviousresearch. No independent-frame or population-level significance claim follows.
