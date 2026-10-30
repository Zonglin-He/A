# R9 · Fixed-q0 geometry and regression-head diagnostics

**Completed real experiments: original target selection held fixed, genuine normal pseudo updates, separately labelled GT controls, and a post-score no-GT optimizer bridge.** The head can make substantial coordinate corrections under correct supervision. The original finite-step rule strongly underfits even the existing pseudo target; fitting that same target recovers its small direct-use benefit. Current G is still weak, provides no size supervision, and does not outperform its same-information direct output. No new TTA method is adopted.

## What was actually executed

The source is the audited Waymo-only epoch100 model,100epochs/15300sourceupdates, checkpointSHA0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd. One3Dgrounder, no teacher/EMA or source retraining. Fresh strictFP32 forward with the same point/text conventions as R8. Normal11arms all use the originalq0; fixed-q0G never chooses another query. Sameforward sourcefeatures are sealed and reused by the GT13controls and pseudo3bridge, so no cross-process query-slot transfer occurs.

Five scenes each contribute32differentframes, hash-selected without GT/color/oldscore and excluding60previoussmoke frames: Drone64/2scenes,Quad96/3scenes. This is a scene-balanced diagnostic, not full5912population evaluation or fresh confirmation; targetVAL had priorR5/R7/R8research exposure. R8full andR9cohort have different denominators and sourceforwards: do not compare their rawmeans to claim a measured recovery of R8associationloss. Holdingq0 removes that path structurally.

Normal11predictions and160real sourcecaches were sealed before GT release. GTcenter/size targets are then built into a separate strict{id,target6_modelcoords} index; noGTyaw/class/identity enters theGPU. ControlledGT branch never callsG/RGB/P/backbone. The exact retained count is160×(11+13+3)=4320armoutputs, with1760normal,2080labelledcontrol and480exploratorybridge outputs retained.

## Fixed-q0 normal results and the no-GT optimizer bridge

All entries are mIoU percentages on exactly the same diagnostic inputs.

| Branch | Drone n64 | Quad n96 | Role |
|---|---:|---:|---|
| Frozen source | 11.784652 | 29.447018 | Unlabelled normal branch |
| RGB Direct G, no update | 11.868878 | 29.501580 | Unlabelled normal branch |
| RGB normal parameter1 | 11.785441 | 29.448009 | Unlabelled normal branch |
| RGB normal parameter3 | 11.786943 | 29.449936 | Unlabelled normal branch |
| RGB same-pseudo one-step fit | 11.868878 | 29.501580 | Post-normal-score exploratory optimizer check |

RGB DirectG−source is+.0842255pp/+.0545620pp. Normalparameter3−source is only+.00229081pp/+.00291735pp. Acc25 remains13/64 and47/96; Acc50 remains5/64 and31/96 for source, RGBdirect, parameter3 andpseudoexactfit. The same-information direct output reproduces the fitted model benefit; this is not a unique parameter-adaptation gain.

Actual RGB nonzero pseudotarget/gradient/parameter/outputchanges are11/64 and19/96; zeros53/77 remain in all means. PC-only has1/64 and2/96, and its direct mIoU changes are approximately+.000309pp/−.001923pp. NormalRGB direct andparameter3 signedIoU counts are4gains/2harms/58ties and12/2/82 at tolerance1e-8. Raw unshrunkconsensus directions are37positive/17negative out of54defined onDrone and44/52out of96 onQuad; the unaccepted and adverse signals remain visible.

The new optimizer bridge reads only sealed normal directG targets and sourcecaches, not GT, scores, media or newly computedG. A real one-step center L2 backward/SGD step with inverse self-feature-kernel LR fits the existing target. Max coordinate residual to directG is1.90735e-6m; RGB relative residual medians6.03e-6/2.37e-5 among11/19nonzero targets. Every zero-target case has zero effective parameter andoutputchange, even though the code executes a step.

This bridge was registered after normal results were visible, under the user-authorized branch of checking optimization when directG is better. The algorithm is an analytic target-fit rule with no LRsearch or newG, but results remain exploratory. It cannot be called heldout confirmation or evidence that parameter updating exceeds directuse.

## Why the old three steps were weak

For one fixed linear regression output p_j with frozen features h and self-kernel Khead=||h||²+1, the registered normalized SmoothL1 loss gives:

`Δp_j = −αβ · (Khead/Kmax) · (w_j/Σw) · clip((p_j−target_j)/β, −1, 1)`

Hereα=.25,β=.1m andKmax=max(Kcenter,Ksize). Equal center-only weights divide by3; equal center+size weights divide by6. In exact arithmetic withKhead=Kmax, each equal-weight center-only coordinate can move at most8.333mm perstep. Although normal size supervision iszero, the unused size-feature norm can still dominateKmax and suppress the active center step. Median Kcenter/Kmax is.305954/.261887 on the fullDrone/Quadcohorts.

| Nonzero RGB target cases | Drone11 | Quad19 |
|---|---:|---:|
| Median projection of parameter3 displacement onto full pseudo correction |5.4065%|5.8447%|
| Median relative residual still left after parameter3 |94.5935%|94.1553%|
| Median nonzero pseudo center displacement |12.422mm|2.419mm|

Independent step verification retains FP32 rounding: maximum excess over the exact-arithmetic bound3.274e-6m andmaximum analyticstep discrepancy4.419e-6m. This is a concrete finite-step/scale limitation. The exact-fit comparison changes both loss and per-head step rule, so it does not isolate LR alone. Removing this optimization attenuation recovers the existing pseudo target; it does not improve that target, fix an incorrectly chosen source instance or establish cross-input generalization. A singlequery bias can fit arbitrary coordinates, so exactfit is an engineering capability/control rather than a proof of learnable domain transfer.

## Labelled GT center and size controls — NOT TTA results

Predeclared spatialconfirmation requires targetIoU>.25 andmargin>.1 over at leastoneunique annotated non-target, after deduplicating repeated target/others boxes within theframe. The inventory has no stable instanceIDs and may be incomplete. In this cohort the condition exactly equals the sourceAcc25-success set:13Drone/47Quad. It is a GT-conditioned geometry subset, not independent semantic identity proof or a population sample.

The table below is restricted to that same subset. All GTcontrols use identical normal cachedfeatures andq0, and nativeyaw stayszero.

| Labelled diagnostic | Drone n13 mIoU% | Quad n47 mIoU% |
|---|---:|---:|
| Source | 47.325752 | 56.992294 |
| GT center direct | 61.918503 | 64.123190 |
| GT size direct | 48.523754 | 62.691867 |
| GT center+size direct | 67.428812 | 76.819647 |
| GT center+size normalSGD3 | 47.843287 | 57.991370 |
| GT center+size real exactfit | 67.428797 | 76.819636 |

Center-only GT correction has the larger mean effect inDrone; both center andsize matter inQuad, with nonlinearIoU interactions. CurrentG leaves sizeunchanged, so it does not test the sizeadaptation component highlighted by Reg-TTA3D. GTlocaldimensions atnativeyaw0 are not a maximumIoU upperbound; the residual difference from a perfectGTbox includes the unmodelled yaw.

On the spatial subset, source center/size L2means are.543060/.281695m inDrone and.177854/.213037m inQuad. GT-CS normalSGD3 leaves.538185/.266438m and.173972/.197748m. Actual exactfit reduces them to roughly1e-7–1e-6m. Across all160cases, C/S/CS realnonzero gradients andparameterchanges are verified; inactivecoordinates andunsupervisedheads remain sourceexact. Maximum supervised coordinate errors are1.14441e-5m/3.81470e-6m, below the fixed5e-5m criterion. Allstate restoration/cache/q0 checks pass.

The head can therefore realize large coordinate corrections, while the old finite-step rule realizes only a small fraction even withGT. These controls do not show that an unlabeledG can know those coordinates or that per-exampleGTfitting generalizes.

## Human image review and identity/error separation

Root actually viewed all20fixedhash figures,2spatial-confirmed and2unconfirmed per scene. Fullcaption,source/directq0 and rotatedGT7D overlays were visible. This was not blinded to annotation/spatialstatus and did not affect any prediction, Gthreshold or optimization choice.

The qualitative labels are12same-visible-target-plausible (6Drone/6Quad),3different-visible-object,4source-off-target/outside-visible-region,1occluded-ambiguous. These are visual assessments, not authoritative3DinstanceIDs or estimated population error rates. Two spatially unconfirmed cars still plausibly correspond to the correct visible car with center/yaw/extent errors. Other unconfirmed examples clearly address anotherperson/car or background. Thus lowIoU cannot be equated with instance-selection error.

On the12human-plausible cases, the following sealed-row coordinate/GTdiagnostics are descriptive only:

| Human visual subset | Drone n6 | Quad n6 |
|---|---:|---:|
| Source center L2 m | 1.086125 | 0.293384 |
| Source size L2 m | 0.197111 | 0.260968 |
| Source mIoU % | 37.649169 | 50.046453 |
| GT center direct mIoU % | 56.806787 | 55.549228 |
| GT size direct mIoU % | 37.595092 | 55.395728 |
| GT center+size direct mIoU % | 59.697345 | 68.904206 |

The humanQuad subset also retains a small normalRGBDirectG task decline,50.046453%→50.042519%, despite slightly lower centererror; centererror reduction doesnot guarantee IoU gain. The humanDrone subset also retains a slight negative GTsize-only effect; do not omit it or claim all size corrections help. This is consistent withcenter/size/yaw interaction andtiny sample limits, not evidence against sizeadaptation in general. Private figures/captions/coordinates/IDs remain local; public materials retain only aggregatecounts/errors and anonymouscase scalars.

## Audits, failure lineage and cost

Normal160 sourcecaches and11arm contracts were independently checked before GT release. Normal scorer/audits validate sameforward inputhashes, fixedq0, source/CLIPfreeze, all denominators and repeatedexpression dedup. Independent reduction validates1340normal/cancellationcells,364GTsummarycells,180bridgecells and coordinateerrors from actual sealed arrays. AllC/S/CS exactfits and the noGTbridge pass the registered coordinatechecks.

The first rootnormal launch passed an incorrectCLIplanhash and was rejected at the first hash gate before model/data/outputcreation; the log/receipt remain. Actualnormal execution isattempt002 with the original unchanged plan. The first GTauditor draft compared serializedfloat32 residuals using float64 subtraction; its failure andoriginalcode remain inhistory. The correction uses exactly the declared producerfloat32 arithmetic, with no prediction/scorer/tolerance change.

Normalexecution sealed pipeline walltime is58.355s; GT13controlledexecution153.712s; the noGToptimizerbridge217.969s. These include source model loading/replay, fullstatehashing, diagnostictraces andallmethodarms, not matched deploymentlatency. No fullsource retraining was performed. AllR9GPUprocesses exited and no nextGPUrun is queued.

## Decision

**Do not reject the regression-adaptation route on the basis of R8/R9.** R9 supports large supervised coordinate headcapacity, weak positive fixed-q0G geometry on this diagnosticcohort, and substantial finite-step optimizer underfitting. It doesnot support a unique unsupervised parameter-adaptation contribution: the correctedfitter matches the same-information direct pseudo output, improvements are small, Acc25/50 unchanged, only2/3sceneclusters, and no size pseudo supervision exists.

**No new TTA method is adopted.** Keep sourceq0 as the frozen reference, retain directG and exactfitting as diagnostic controls, and require stronger independent center/size pseudo supervision before expanding online updates. Do not reintroduce targetreselection into a regression-only comparison or assume unconfirmedIoU means wronginstance. The current singlemodel/noEMAconstraint remains. No more parameter sweep, source training or automatic continuation is scheduled.
