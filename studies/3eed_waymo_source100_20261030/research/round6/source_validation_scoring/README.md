# SourceWaymoVAL CPU scoring, epoch100 primary

This is a source-only post-seal scorer for the exact two-head NPZ API in `crossplatform/round6/source_validation.py` (frozen SHAd428c5c566ee85748e843131879572195dcc106de0749ca203a35110281c5b22). It imports no torch/model/GPU code, performs no optimization, and never loads a mixed checkpoint or a Drone/Quad index. The approvedWaymoVALprivateindex supplies the actual original loader order and independently validated float32 GT7D. The training source labels are legitimate; prediction sealing is a transaction boundary, not a claim those labels were never seen historically or during source-supervised monitoring.

Only code and synthetic fixtures have been run so far. Real scoring requires the parent to write an explicit `CPU_source_val_scoring_authorized:true` release after epoch10/20/.../100 prediction sealing. Epoch100 remains the fixed primary, even if an earlier sourceVALscore is higher. A best-sourceVAL checkpoint can only be separately labeled secondary by the parent; no target-domain score influences this code.

## Call

```
<workspace>/crossplatform/env/bin/python <workspace>/research/round6/source_validation_scoring/score_source_val.py --predictions EPOCH_SOURCE_VAL_PREDICTION_DIRECTORY --checkpoint IMMUTABLE_EPOCH_CHECKPOINT --release PARENT_SOURCE_VAL_CPU_RELEASE.json --out NEW_SOURCE_VAL_SCORING_DIRECTORY
```

Use the actual per-epoch immutable checkpoint, not a changing`last`pointer. The scorer verifies checkpoint/prediction/index/program hashes before loading sourceGT. It rejects duplicate/missing rows, wrong head order, a changed privateindex, non-source metadata paths, wrong token lengths, changed model/buffer flags, nonnativeTop10 order, inconsistent raw/clamped boxes, nonfinite values and any independentselected3DIoU/threshold disagreement. Failures are raised and partial outputs are retained; no case is silently skipped.

## Release fields

`RELEASE_TEMPLATE.json` lists the fields the parent must bind. It is intentionally not authorized. Fill the current epoch10–100, sourcepredictionseal/checkpoint/modelstate/order/index hashes and this programlock hash, then setCPUauthorizationtrue. The release must keep`source_only:true` and`primary_epoch:100`. The prediction seal comes from the actual source-validation writer, not the old round5frozenunionbaseline.

## Numerical definitions and outputs

The official readouts use the released sourceGTclass-dependent positive_map. Text readouts use the first fixedlexiconnoun from the exact same fullcaption forward; no secondnoun-only inference is performed. Both BBF/BBS atlast/proposal are scored separately. NativeGPUdescendingTop10indices are retained, including ties; no CPUresort changes the winner. The model used originalB24dynamicpadding, and the scorer verifies `T_batch`, independent actualtokenlength and the final14-row batch instead of replacing the denominator by B1.

Predictedrawdimensions are preserved and checked against the officialminimum1e-6clamp. Nativeprediction yaw is0. GT is actualfloat32postfrontrotation data promoted tofloat64 before geometry. The main batchedIoU is rectangleintersection×verticaloverlap. A separate general3Dconvexintersection computes every unique Top1selected box from the eighthead/readout arms, using float64 extents before both corners andvolume. It must agree to1e-8and produce exactly the same strictIoU>.25/.5decisions. Full256proposal coverage is computed by the primarybatchroutine; it is not claimed independently rescored by3D.

`RESULTS_PRIVATE.json` contains all3782rows, each arm's selectedquery/IoU/nativeTop10retrieval metrics, GT-assisted pooloracle, invaliddimension counts and actualtokenlength. `summaries` has samplemicro/equal-scene mIoU, strictAcc25/50Top1/5/10 and GT-assistedTop256 coverage. SourceGT-assisted oracle retrieval is not an online model output. Last/proposal mIoU is not the original sevenheadsharedlegacyaccumulator. This cache deliberately has onlytwoheads, so it does not fabricate a sevenheadlegacymean.

`SCORING_TRANSACTION.json` binds the verified input hashes; `SEALED.json` binds all outputfiles and epoch/checkpoint. Every perquery all-candidateIoUvector stays private. Public export can project `summaries`, `invalid_counts`, `independent_selected_audit`, epoch/checkpoint hashes and timing, but must not copy the originalsource scene/frame/objectindex or privateGTindex.

## Synthetic checks and provenance

`check_synthetic.py` has no dataset/GT/model access. It checks exact.25/.5 boundaries, Top1/5/10/256 definitions, preserved nonstable native tieorder, malformedinput failures, rawnegative/zero dimensions and exactclamp, float32→float64promotion, plus128seeded randomgeometry comparisons between the independentgeneral3D and batched2Dimplementations. No synthetic value is a model performance score.

`batch_yaw_iou.py` is a byte-identical copy of the previously audited pure geometry module in`research/round5/scoring`; `obb_iou.py` is a byte-identical copy of the general3D puregeometry module in`tools`. Neither copied module reads a dataset. Their source hashes and the source-data audit are bound in `SCORER_PROGRAM_LOCK.json`. No old round5driver with target/mixedcheckpoint paths is imported.
