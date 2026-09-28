# Large-Magnitude Mask Control — completed / independently audited

Source training/development oracle, single fixed scale from2 selected prior source endpoints, descriptive uncorrected paired16-parent bootstrap; no new target/generalization claim.

Only magnitude changed. B1 residual and original late masks retained; each correct/wrong delta independently normalized to stock F. Event .087687, spatial .170316. Three predeclared zero event directions remain neutral; all16 kept, eligible13 also reported. No optimizer.

|Condition|tIoU %|sIoU %|vIoU %|valid formats|
|---|---:|---:|---:|---:|
|original|54.003027|61.929737|41.113359|16/16|
|event_large_correct|51.471027|60.637670|39.685633|16/16|
|event_large_wrong|49.107297|59.441176|38.295906|16/16|
|spatial_large_correct|52.173759|60.217089|40.361538|15/16|
|spatial_large_wrong|52.173759|59.800988|40.324074|15/16|

Primary changes relative to B1 (pp):
- event: correct-minus-base -2.532000, CI[-11.08444353593744, 3.9610271261475782]; correct-minus-wrong +2.909207, CI[0.0, 8.400830831962592] (13 parents).
- spatial: correct-minus-base -1.712647, CI[-7.971288556230801, 2.311244973001488]; correct-minus-wrong +0.416101, CI[-0.4831580050203457, 1.520431798685832] (16 parents).

Temporal correct/wrong changes intervals in6/5 cases; correct-minus-wrong is positive in2/13 and zero in11. One positive difference is genuine local benefit; the larger difference is less damage than a severely harmed wrong control, not improvement overBase. Correct temporal loses2 parents by>5pp vIoU and retains6/7 native-good v,7/8 native-good t.
Spatial has7 positive/8 negative/1zero versusBase; wrong has6/9/1. Correct-minus-wrong s is9positive/5negative/2zero and CI crosses0. Each spatial arm has1>5pp v loss. Both retain7/7 v and8/8 t native-good threshold cases, while continuous harms remain.
Both spatial arms on anonymousP02 emit <null> instead of <|box_end|> at the last box-block structure position. All raw blocks remain saved; the unchanged official whole-output grammar makes all three metrics zero. Spatial event logits/reference/interval are identical toBase on all16, and direct endpoint-only t mean remains54.003027%. Do not describe official t decline as a temporal-path change.
Previously selected temporal reachability caseP07 remains29.166667% tIoU under both large masks, versus68.055556% from the saved optimized span direction at the same scale. Spatial caseP03 improves59.654986→64.494721% s under correct large mask, versus73.834416% in the saved span control; wrong large reaches64.081856%. These two outcome-exposed examples are diagnostic, not a generalization test.

Engineering: original large_mask001 stopped at CPU/CUDA comparison after2 saved native outputs,12.035057106s. Isolatedv2 changes only device-neutral readback; CONFIG/INPUTS exact and both partial outputs replay exactly. Newrun155.026535067s; all failures/loading/replay/finalization retained.7CPU controls,240 scalar/tensor geometry (max3.331e-16),4072 margins (error0),90 independent parent/CI/tail reductions (max8.882e-16) pass. Full stock/FP32 endpoints+scaled deltas reconstruct256 BF16 before/after endpoint hashes exactly; realized ratio error<=6.180e-9. No raw/weights/labels published.

Decision: the fixed strong intervention actually reaches PTD, but increasing magnitude alone does not establish a beneficial privileged policy. This is a mixed result, not wholesale collapse and not proof that all scalar masks or3D adaptation are impossible. Small source-specific correctness signals are retained. Reachability is supported only in the two earlier source controls; evidence directionality does not pass the current teacher gate; actual-reader learnability and distillability remain untested.
Next proposed factor, if pursued: context support with this operator/location/magnitude frozen and matched wrong evidence. A distinct protocol must resolve context support and negative matching before any run; no context/GPU registration, reader training, external expert or OPD was added in this experiment. No scale/LR/precision/step sweep. Large equivalent latent coefficients alone are not a stop rule.

Settled cumulative GPU allocation: 42340.06753310408s, cap=null. Source-only oracle; old target development exposure, all optimizer corrections, cancelled full-source training and CURRENT are preserved.
