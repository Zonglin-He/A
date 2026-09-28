> Historical v3 source-preparation protocol. The user cancelled the full-source fit at35 committed steps; do not automatically resume. See docs/desta3d_v3/EXTERNAL_PRIVILEGED_OPD.md.

# DESTA-3D: full source preparation and qualified structured adaptation

Status: authorized by attachment 6e648ae6 on 2026-09-28; implementation/intake in progress. This is a new source-preparation route, not a promotion of B1 or a claim of mature representation.

## Evidence and question

The valid v2 B1 integration checkpoint used 155 of a 775-update schedule on 618 queries/95 Vid parents. Its source-val difference from matched Frozen was +0.0088 pp with a CI spanning zero. The source16 FP32-final-head control worsened native vIoU by 4.0996 pp and had two >5 pp negative tails despite local descent. These results, all prior positive/negative outputs, and the integer-optimizer-key incident remain intact. Undertraining, conditional supervision/readout differences, and unreliable unlabeled signals remain competing explanations. Full-source training is a new condition to test, not an established cure.

## A. Full source preparation

- Use all queries (captions and questions) from official VidSTG train and all 4,500 official HC-STVG1 train clips. HC1 train/test have disjoint raw-parent IDs; HC2 train/val share 236 parents, so HC2 is not silently substituted for this experiment. Read official train annotations for supervision; benchmark test/val files are inspected for identity metadata only. Existing historical exposure remains disclosed.
- Hash-rank raw parents independently per domain (namespace `desta3d-v3-source-v1`, seed 20260928), hold out ceil(10%) for source validation. No query from a parent can cross the split. Held-out source is development data, not claimed historically unseen. Vid official validation/test and HC1 official test remain outside fitting. Old HC2 benchmark panels overlapping HC1 train must not be reported as independent held-out tests of this new mixed-source model.
- Inventory archives and verify extracted HC1 bytes against the original split ZIP/tgz, not same-name HC2 files. Decode/check media before including it in the final runnable training manifest. Missing/corrupt inputs get explicit records; no score-based replacements. Vid archives support on-demand CRC-checked extraction with a bounded disposable media cache. Retain original archives and scientific evidence.
- Preserve hidden128 dual readers, official frozen PTD4B, two independent branch injections/prefills, P1/offset/joint disabled. A new training registration will lock initialization, complete eligible roster and horizon after intake; never resume old optimizer snapshots.
- Implement explicit GT-conditioned structured PTD MTP endpoint and coordinate objectives. Endpoint loss is the sum of two CE terms over the actual T learned time tokens; box loss is the mean over frames and four coordinates on the complete 1001-token coordinate support. Semantic/structural/null tokens are not endpoints/coordinates. Keep a small explicit original branch CE regularizer and referent/event BCE. GT-conditioned official training prefixes are NOT free native cached decoding; native positive controls below will test that gap.
- Starting recipe: reader/input/query/FiLM/LN LR 3e-5; evidence-head/output/gate LR 1e-4, AdamW wd0 clip1 accum4 (actual last-window divisor), lambda_M=lambda_E=.1, lambda_original_PTD=.1 (mean of two original branch CE terms). Original BF16 PTD/head remains; the negative FP32 control is not silently made default. Exact scientific values are locked before training, not tuned against target outcomes.
- Domain-balanced complete epochs: shuffle every Vid query once and cycle shuffled HC training queries to the same count, interleave Vid/HC, disclose HC repetition. Scheduler derives from this actual epoch length and max 5 complete epochs, 5% warmup over the declared horizon; source-val patience2 after at least 3 completed epochs. Never call 775 updates a full epoch on the new large roster. Clean plus generic mild views only; exact target noise/blur severities are not source augmentation. Detailed augmentation registration precedes fit.
- Full per-epoch source-val predictions and matched Frozen at identical physical pixels are sealed before source-val scoring. Parent macro v/s/t, paired parent CI, domain results, all failures, retention and >5 pp tails; selection only by predeclared source-val rule. Integration candidate requires source clean mean >= matched Frozen, with CI/tails disclosed, not a stable-gain claim.

## B. Native actuation qualification (after A)

Run actual native time endpoint GT positive control and spatial coordinate positive control at each sample's fixed native reference/time. Map every supported physical frame to source labels; missing GT support remains explicit. No target labels. Start with branch FiLM + branch LN only; shared LN frozen. If unsupported by this positive control, separately register +branch output/gate (post-gate loss required), then the last branch reader pointwise layer; shared representation is last. Different branches may qualify at different scopes. No automatic conclusion that a late adapter is necessary from a failed scope.

The detached spatial-referent weighted event pooling variant is a separate factor, initially OFF. It uses softmax_xy(stopgrad(M)/tau) with fixed tau1 and pools event features, retaining independent residual pathways. Source evidence/task comparisons must precede adopting it.

## C/D. Unlabeled TTA and evidence-privileged branch Self-OPD (conditional)

Only after source preparation and native interface qualification, test whether source-corruption unlabeled updates aid the original task. Labels are offline diagnostics, never unlabeled update inputs or online state selection. Maintain no-TTA/TTA, native-good and negative-tail comparisons.

An evidence-view teacher uses the same model and same observed corrupted video; no clean counterpart or new teacher model. Temporal views preserve frame IDs/times, spatial views preserve full coordinates. Teacher quality must first be demonstrated on source data. Shared student rollout tokens, branch-specific endpoint/coordinate supports, source-frozen reliability rules, and explicit skipped-support accounting precede OPD. Wasserstein/geometric objectives are deferred separate factors. Video-OPD/Vision-OPD motivate hypotheses; this is not their reproduction or a validated novelty claim.

## Engineering and reporting

GPU serial; cumulative cap=null, prior measured total 38820.55116519004 s. Every load/failure/replay and nonoverlapping wrapper interval counted. Minimum free disk 8 GiB, bounded stages only engineering review. New files live under `artifacts/desta3d_v3`; v2 pins/results, CURRENT, old queues untouched. No automatic timer recreation, GitHub push or production promotion. An early B1 interface test is implementation validation only, not the post-full-source positive control.
