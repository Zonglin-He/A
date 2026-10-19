# Crop evidence to E propagation: cached32 mechanism audit

2026-10-19. Completed descriptive cached-evidence analysis / audited. No new predictions, model or expert calls, training, student updates, GT-annotation reads, or task scoring. This is not a new OPD experiment. Sources are the sealed observation_factorial32 NH16/NV16 development run and identified historical controls; one query per parent. Original cross-domain frozen EMA TA-STVG, DINO-T, four observation positions and E+finalMBR remain unchanged. Formal FH/FV64 is not used.

## Decision

The direct S output is a negative E readout result, not an OPD failure. The cache identifies a real loss of sparse localization evidence, at different stages by dataset. HC loses useful selected references during constant-correction propagation. Vid has a large selection failure concentrated in NV016, plus one useful accepted reference rejected by E. These findings justify considering a different cross-frame state/supervision interface; they do not establish a deployable 2–3pp gain or the necessity of Live refresh.

Do not resume the old fixed-box-IoU reward loop, global query residual, geometric reference reseeding, temperature/step/crop-size sweep or new model training. No experiment is automatically authorized by this report.

## Matched-support decomposition

For the SAME known frame with an admitted crop and a selected reference, let E be original E quality, D raw crop top quality, R the reference actually selected by E, and S its propagated output quality. The exact descriptive identity is S−E=(D−E)+(R−D)+(S−R). It is NOT a controlled causal ablation; selection and propagation can interact.

We first average those differences within each query, then across supported parents; intervals use parent bootstrap10000/seed20261018. This conditional anchor diagnostic is not complete-cohort vIoU, not unseen-frame quality and not the all16 denominator.

| Stage, anchor IoU pp | HC:47 frames/15 parents | Vid:15 frames/7 parents |
|---|---:|---:|
| Crop top D−E | +3.3990 [+.3334,+6.9213] | +.7608 [−2.0654,+3.5703] |
| Selection R−D | +.3022 [0,+.9066] | −5.2272 [−15.6815,0] |
| Propagation S−R | −3.9794 [−7.2452,−1.2003] | −.9974 [−3.7107,+1.8469] |
| Final S−E on this same support | −.2782 [−.8686,+.2977] | −5.4638 [−16.3147,−.0018] |

HC:28 raw-top gains, zero of those reversed by selection;29 selected-reference gains,13 of those no longer above E after propagation. The 13/29 count is not a majority-of-parents test. Vid:8 raw-top gains,2 reversed by selection;6 selected-reference gains,3 lost after propagation. All frame signs and supported parents are retained in CHAIN_PUBLIC.json, not treated as independent trials.

Support caveat:Vid has16 known admitted crop frames; E rejects one of them, NV005/position6. Its saved E/crop-top/propagated qualities are .409422/.879292/.399017. That +46.9870pp local crop opportunity is retained explicitly but cannot enter the R−D−S telescope because R is missing. Including it gives the older accepted-crop-vs-E diagnostic +2.1737pp; excluding it for common support gives +.7608pp. These are different masks, not conflicting results or a changed deployment gate. HC has no such known admitted reference rejection.

## What the implementation actually does

`vg_tta/stvg_dino_supported_correction32_v1.py:estimate` searches finite candidate/outlier assignments for a global minimum of the truncated standardized residual fit plus relative target-score cost. Candidate evidence is expressed relative to Native Source, in z=(cx,cy,logw,logh), not relative to the final E box. Selected reference residuals are averaged into ONE four-vector. `apply` adds that vector to every Source frame.

Therefore, for every actual frame t,

    z(S_t) − z(E_t) = delta_crop − delta_old,

which is independent of t. Direct cache checks verify that identity across all32 queries, max numerical error4.44e−16. This is a four-dimensional correction family; it cannot retain different correction magnitudes/directions at individual anchors. It also changes unobserved frames without reading those frames anew. Inconsistency of reference residuals is a geometric fact, not proof that any particular reference is correct or that an unconstrained trajectory would improve.

The frame admission gate and final candidate choice are distinct. Among60 selected HC crop references,4 are non-top and below the .35 frame-admission score; among33 selected Vid crop references,6 are non-top and below .35. This is permitted by the existing protocol, not a broken threshold implementation. Raising a threshold is not the proposed remedy.

## Two distinct concrete breakpoints

HC NH011/position23: E=.301612, raw crop top=selected reference=.846459, propagated S=.294434. Correct local evidence is retained in selection but not reproduced at that anchor by the shared correction. Across the four known anchors NH011 has +15.2361pp raw/selected local gain, −14.9302pp propagation difference and only +.3060pp remaining anchor gain. Saved complete-v change is +.0546pp. NH003 similarly loses8.2821 of9.2011pp mean local opportunity during propagation. NH014 is a retained counterexample:raw crop top is already worse than E by4.1442pp on average; preserving every raw crop would also preserve harmful evidence.

Vid NV016:

| Original node | E IoU | Crop top IoU | E-selected reference IoU | Propagated S IoU |
|---|---:|---:|---:|---:|
|40|.752732|.837349|.377380|.361528|
|50|.806549|.581506|.340953|.335756|
|61|.605523|.693726|.296549|.328183|

Position29 has unknown GT and stays unknown. At all four positions E selects candidate1 instead of raw top candidate0. Raw-top target scores .8123/.8057/.4040/.8499 become selected scores .0734/.0746/.0762/.0931. For the chosen shared correction, saved objective=26.9571 (geometry17.1810 + score penalty9.7761). The forced-all-top residual-mean point has truncated top-only geometry60.8003 and zero relative-score penalty. This explains why optimizing consistency can prefer these low-score alternatives. The latter number is a descriptive objective at one point, NOT a best top-only-subset optimum, GT-optimal assignment or new predicted trajectory. Original exhaustive audit already established the estimator's optimization correctness.

NV016's raw top is better than E at two of three known positions, but its mean raw-top−E is −1.7408pp, so it is not uniformly correct. Selection then loses another36.5902pp on those known anchors; propagation adds+.3530pp relative to the selected poor references. The complete-v loss−26.1278pp must not be attributed entirely to propagation or detection. No identity/background label is inferred merely from the lower IoU.

## What is and is not established about 2–3pp capacity

The previous S−E complete-v remains−.2194/−1.6290pp and unseen-actual-s−.0748/−3.2841pp (HC/Vid), all16 parents. No new complete-v numbers were computed here. Raw crop versus old full-image raw top still has no established improvement:−2.9525/+1.1457pp with wide intervals. A better local comparison against E does not prove the crop teacher is better than the old detector.

Historical same NH/NV [reference capacity32](../reference_capacity32/REPORT.md) found original-pool full-trajectory oracle gains+3.1763/+2.9676pp; correct-GT-four-reference E gains+6.5058/+10.8284pp. These are GT-assisted capacities, not deployable/crop results. They keep substantial spatial opportunity plausible, but do not identify a reliable selector or prove that new crop evidence supplies it.

The later [E-retaining1792](../e_retaining1792_route32/REPORT.md) correct-reference controls demonstrate actual larger changes, HC GT-Live+4.8221pp full-v and Vid GT-Direct+1.1364pp; real fixed-box-IoU OPD does not realize them. The HC result is compatible with the desired magnitude under correct supervision; the Vid result does not demonstrate a2–3pp realizable gain through that tested student. Do not multiply mean local IoU gain by mean tIoU, sum old/new gains, or treat oracle quality as a prediction of a new method.

Thus HC has a concrete propagation loss worth targeting. Vid first needs reliable preservation/selection of competing object evidence; simply copying every crop top is not supported. Existing caches do not contain predictions from a crop-supervised, frame-conditioned alternative student, nor a second RGB crop after its update. Its full-tube gain and Live−Fixed advantage are not identifiable from these data alone.

## Alternative interface to consider, not executed

The useful change is to keep each observation's localization evidence attached to its own physical frame and candidate mode, instead of making compatibility with a single error vector the admission/propagation criterion.

A concrete candidate is **anchor-specific temporary content states inside the frozen spatial decoder**, before a temporal self-attention step. Distinct observed-frame states can carry distinct local corrections; the already-trained temporal self-attention and per-frame visual cross-attention generate the remaining trajectory. No new network, projection or localization head is trained; all pretrained weights remain frozen, and per-video temporary state resets. Zero state must recover E exactly through the existing paired E-retaining center/log-size map. It is a hypothesis about a different adaptation interface, not a statement that trained self-attention guarantees correct propagation.

Supervise these states using the actual crop/full-image candidate boxes and their retained alternatives at each anchor, preserving whole-box modes. A localization-distribution loss must act on the decoded localization, with no flat all-zero-IoU reward and no conversion back into the old unordered ROI score. Candidate weights remain noisy, uncalibrated scores; preserving alternatives does not solve target identity. Do not silently optimize pretrained LayerNorm or call state fitting network training.

This differs from a shared256D query offset or1792 shared QR/LN update, from replacing geometric reference boxes, and from independent final-center updates. A targeted history search located no matching completed anchor-specific content-state test; this is not an exhaustive novelty claim.

Historical constraints prevent a superficial reopening:

- [Reference reseeding32](../../2026-10-18/reference_seed32/REPORT.md):changing initial Source/E geometry alone barely changed paired decoding (+.0056/−.0003pp full-v). A new name for reference reseeding is not justified.
- [Persistent anchor refresh32](../anchor_refresh32/REPORT.md):reinjecting four geometric references before each layer, with paired E-retaining output, yielded−.0203/−.0031pp full-v; a self-attention path alone is not a reliable transfer interface.
- [Actual latent query OPD32](../realizable_query_opd32/REPORT.md):actual current-state frozen-decoder trajectories were sampled, but the adapted mean was one shared256D query and the reward was fixed-reference IoU. It is not a test of independent anchor content states or refreshed RGB evidence.
- E-retaining1792 correct labels can fit but can also harm unobserved frames. More freedom is not intrinsically safer, and anchor fit must not be a success criterion by itself.

If this candidate is later authorized, one finite decision experiment should compare the same teacher evidence through constant-E propagation versus the frame-conditioned state interface, and compare equal-budget Direct/Fixed/Live within the new interface. Actual Live must refresh student-state-dependent candidates/observations and save their provenance; the current one-view cache cannot establish that. No requirement for hard selection to improve before actual OPD may run. No local parameter sweep, new expert or head. Judge complete-cohort vIoU and unobserved actual sIoU, keep tail harms and require the sought2–3pp magnitude as a measured outcome. If only fixed/direct transfer improves, credit the interface rather than OPD. If even trusted supervision cannot transfer beyond anchors, reject that specific interface rather than iterate cosmetic variants.

## Scope, integrity and cost

All128 saved frame-quality records were retained; the telescope uses62 known admitted/selected frames. Independent audit reconstructs248 stage terms,8 parent-bootstrap summaries,32 uniform transforms and103 input/registry pins. No raw annotation, image/video, model checkpoint or new frame was opened. Torch only loads existing prediction/source/E tensors on CPU, CUDA remains uninitialized. Source checkpoint provenance remains that of observation_factorial32/MODEL_PROVENANCE.json. Exposure, uncertainty and old successes/failures are preserved.

The recorded primary cache-analysis phase is .321s after framework import; audit, review, preparation and publication are not included, so this is not total task latency. All incremental GPU, expert/model, training and student-update counts are0. Original evidence-acquisition costs are excluded. CURRENT/MAIN/PAPER and old paused jobs remain unchanged. Analysis completion does not authorize a new experiment or automatic continuation.
