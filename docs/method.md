# SimplePath + video-wide Native2D feedback

## Fixed computation

1. Run the original frozen TA-STVG Source224 inference unchanged. Its final-layer FP32 start/end logits from both native offsets form the original physical-hull prior q0. Global6 is not a production input.
2. Use the same eight ReferDINO observations and source-calibrated shared geometric initialization. For each sampled frame, retain the nearest past and future observations. The original two-way affinity product and valid axis-affine fit transform each complete expert box. Keep the geometric mass/variance/scale/intersection checks and initialization fallback.
3. Score each candidate by its mean available-anchor appearance cosine. Connect adjacent complete candidate boxes with their IoU. Exact DP maximizes .5 observed mean + .5 supported-unobserved mean + .25 supported adjacent-link mean. Denominators are the original event observed/unobserved/edge counts. Removed: round-trip reliability thresholds and weights, extent RBF, deviation-from-initialization guard, adjacent feature/mask transport. Both geometric directions remain.
4. Run the same SimplePath over the full sampled timeline to obtain an auxiliary trajectory. Pool the original input_proj appearance grid **before text fusion** inside each complete box expanded by the fixed1.5 context. No new representation model is evaluated.
5. Center/L2-normalize the pooled features over the whole video. Native q0 supplies soft event membership. Estimate **one event/background prototype pair from all frames**, both shrunk toward the video's common mean with kappa64. No splitting by offset or cross-fitting remains. The ratio of shifted cosine compatibilities gives per-frame feedback. This is within-video self-consistency with self-inclusion, not calibrated event probability or independent validation.
6. Sum frame feedback over each legal inclusive interval and apply ONE update proportional to q0 times exp(score), beta1. Use the original physical MBR, then SimplePath at the resulting interval.

The original Source's two-offset sampling is unchanged; removing offset-wise prototype fitting does not remove Source offsets. Network parameters, teacher evidence, sampling, precision, seed and fallback rules remain fixed. No parameter search is part of this adoption.

## Evidence and limits

The user explicitly adopts both simplifications after the exposed modular experiments. On the67-query HC availability panel with old time fixed, SimplePath changed full vIoU +.015794pp/unobserved sIoU +.208066pp (CIs include zero), observed sIoU -1.155595pp. The one-pair temporal arm on exposed128/domain changed HC tIoU/vIoU +.641039/+.302139pp and Vid -.052664/-.043750pp relative to the old method. Severe Source-relative t-or-v union increased19→23 in HC and21→25 in Vid; Vid short-event Source-relative tIoU was -.006256pp. These outcomes remain intact. They are modular tests, not evidence for combined full performance.

The new auxiliary path differs on all67 audited inputs. Old pooled temporal features are therefore invalidated. Unchanged geometric candidates and exact Source/E/native logits/eight positions may be reused with input/hash bindings; actual pre-text grids are repooled on the new path. A real cold/cache bridge verifies Source/q0, raw prefix cells, old-pool recovery and coupled new output. No new expert is called for collection.

Each complete official direction must still be evaluated against the fixed query-equal Source+.9pp m_tIoU requirement. The prior HC1902 full failed (+.747072pp); it remains failed. The new method is selected by the user, not retrospectively declared qualified. Full results cannot select parameters.

## Research-only diagnostics and interfaces

The full collector optionally saves propagated-mask candidates for the fixed range analysis using the already computed anchor/frame affinities. These masks never screen, weight or connect the production path. This optional diagnostic work is not presented as necessary deployment cost.

The reusable feedback interface requires spatially supported native local representations and explicit native temporal candidates/priors, with coordinate/time adapters. Feature dimensions and class vocabularies are unrestricted. Only the TA adapter has been verified; cross-backbone effectiveness is untested.

Latest user decision after complete dual-domain results: the measured current method is accepted and the initial Source+0.9 pp per-direction threshold is no longer a work blocker. Original HC point-estimate pass and Vid target_not_met remain unchanged in saved acceptance records. No retuning follows. Current research focus is attribution controls that separate direct expert observations, shared calibration, spatial processing and temporal feedback.


## 2026-10-24: matched TTA384 complete; CG CPU interface preparation

Matched temporal LN Entropy and RGBbrightness*.8 PhotoConsistency finished384episodes/3840updates/16896extraTimeDecoder calls/384newaugmentedSource/0expert; all384sealed before768officialdensechecksmax1.22e-15. AllSource arrays/metrics and reset traces exact; all384arrays/traces/photo contexts readbackSHA. Confirm128/domain Entropy-Source vHC-.217783/Vid-.448877pp (VidCI[-1.043200,-.006582]); PhotoConsistency-.034903/+.037199pp CIbothcross0. Bothobjectivesdecrease384/384, not evidenceof universalTTAfailure; restrictedfinalpasstimeLN, sourceboxes fixed, not canonicalTent/CoTTA. Allscreen/confirm signed, severet/v/uniononce, timegroups, costs public. Mainuntouched. GenuinejointKbudget startedafterTTAexit; shiftsnext,CGlast.

CGpreparation verified58officialsource/configbytes againstpinnedHengLan/CGSTVGf062d84 and2officialweighthashes. SixCPUinterfacegroups: originaltestsampling10fixtures, nativeMAP3fixtures/posteriornumericreference, singlegridpoolmathbyteexact, evaldeadGTreadbypass/restoration. NoCGmodelconstructed/forward/GPU. PlannedCGnative420/singlegrid/nativeMAP plusSourceMBR control distinguishesreadout; archivedCG224/448/6layeradapterisnotreused. Realcheckpointbridge/inference/scoringstillpendingAFTERallTA.


## 2026-10-24: CG native runtime frozen before model execution

Independent CG420/single-grid runtime f767b3424b3a2a11f2f5358a36bbc614b62a2fc3af3dce29be8828052faece5d now implements strict cross-source EMA loading, native pretext/postfusion grids, originalMAP+MBR separation, unchangedSimplePath/Native2D feedback, real first2/domain no-op bridges and independent originalCG CPU metrics. Four further CPU contract groups passed, including official resize/normalization byteexact, no-information/zero-correction identity and6officialCG-vsDense synthetic checks error0. No actual CG checkpoint model/forward/expert/realRGB/GT access yet. Hard gate rejects model construction until allTA inclshift confirmedsealed/scored and ownedexit. OldCG224/448/6layer/optimization not used. UnfrozenCPU roundoff/metric-signature failures preserved and repaired before freeze; no science/model run. Native420FP32 resource/true no-op remains pending and may fail; no silent precision/resolution/frame changes allowed. Budget/scheduleGPU worker continues unchanged; no duplicate or CG worker launched.


## 2026-10-24: exact K32 search limit preserved; independent budget control-flow recovery

Original budget screening stopped after82completequeries plus8arms ofquery83:1,074boundpredictions preserved and everyreceipt/predictionSHA checked. On Hybrid32 of the next Vid query(T200), exactconsensus exceeded predeclared2millionnodes and returned no approximate answer. Oldcontroller/predictor/teacher and waiting shiftqueue exited; sharedlockfree. Preserve alloriginalsource/plan/log/partialoutputs; no oldstage seal/score. Budgetv2 fpc5c24f739ebd160f8b252ddf5b9539edc6e2beda9cf3215c7532656d320dc422 only catches precisely that resource error perconfiguration, records missingprediction and continues original13configs. No threshold/K/seed/math/evidence/accuracyselection change. Reverse-edits recover originalbudget.py byteforbyte and threecore helper ASTs identical. All1,074oldpredictions migrated exactbytes/newreceipts with originalcostprovenance;0extraexpert for migration; failingnativeRGBcontext/cacheSHA verified andfailure preservedwithoutrepeatsearch. New queue resumed beyondquery83, oldcompletedexpertwork notrerun. NativeHybrid8 originalFull no-op still enforced; incompleteHybrid32 cannot enterconfirm by unchanged predeclaredtwo-domain completeness+10xGPUcostrule.

Same frozen shiftcode/plan requeued via dependency-only queue_shifts_v3 after newbudgetconfirm receipt; originalfailedwaitingqueue retained. CGv2 plan160febd9f0962e09c30c7dc8cd348dda1e0b45b12e5a47ce55a85dc6759e2621 only rebinds recoveredbudgetreceipt/newownedPIDnames andoutputnamespace; nativeSource/feedback/scorer same. FourCPUcontractgroups repassed;0CGmodel/forward/RGB/GT, realbridge LASTpending. v1CGpreparedcode/plan preservedunused, no oldreceipt spoofing. Budgetfigures executable code now prepared, gated on fullstage seal+score; no chart/data invented while screening continues.
