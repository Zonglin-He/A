# Round5 official-protocol audit v2: raw-extra-field clarification

Audited upstream commit: `843a901f5367a6be5cc484b97c6ecad462961dfa`. `src/grounding_evaluator.py`, `utils/eval_det.py` and `train_dist_mod.py` have no local diff. Dataset differences are removal of unused `cgi`/Open3D imports. This review reads implementation, not new target labels, and does not invoke a model or GPU.

## Official eligible target and span

`load_3eed_annos` enumerates every `ground_info` object in every available frame from the platform VAL split. The sample identity must contain platform, relative metadata path and object index; frame alone would merge multiple targets. Missing sequence directories are silently skipped by the original loader, whereas missing metadata asserts. Full-VAL reporting therefore needs raw split/sequence/frame/object counts and explicit missing-input/error counters rather than an implicit success-only denominator.

Eligibility and the evaluated span use the **GT class**. The code loops the class's platform-specific synonym list in list order and appends every case-insensitive whole-word match. It takes `positions[0]`: this is synonym-priority first, then first occurrence of that synonym, not the globally earliest noun in the caption. If no synonym matches, it skips the target. `get_positive_map` maps formatted-caption character offsets using the local RoBERTa tokenizer, normalizes the token mask, then both evaluator modes binarize all positive values back to one. A class match whose char-to-token map becomes empty remains a retained all-zero-score/tied-ranking case; it must not silently become another skip.

This is a label-assisted evaluation readout, not unconditional language-only deployment. To preserve prediction-before-label separation for the new run, compute and seal scores for all six possible class spans before choosing the true class column offline. Eligibility for all six classes is likewise GT-free. Raw input enumeration may open containers containing annotations to extract captions/object indices, but should not read or feed class/box/others values into source forward. The post-seal scorer alone chooses GT class and target geometry.

The official caption formatter adds a leading/trailing space and spaces before commas for span tokenization. Model input instead normalizes whitespace/commas and appends ` . not mentioned`. These historical details should be copied for parity rather than silently replaced by new offsets.

## Prediction geometry and metrics

Each head predicts 256 center/size boxes, with six values `[x,y,z,l,w,h]`; the evaluated predicted yaw is zero. No PCA, point refit, GT-yaw injection or adaptation belongs to the official primary. Target geometry follows the dataset transform: Drone z+1.8m, Waymo suffix-dependent XY rotation and yaw addition, Quad unchanged. Raw Drone/Quad annotations can have nine values, but the actual dataset creates seven-value GT buffers from `bbox[:7]`. The final two raw fields remain **unnamed extras**: their generation schema has not been established, and the rotation labels in a generic utility docstring do not establish their meaning. Only the first seven geometric fields enter the exercised official scoring pipeline. The final GT tensor is float32.

BBS uses softmax over its entire fixed classifier dimension and sums target-span tokens. BBF uses query/token dot products divided by .07, softmax over **all actual language positions**, then sums target-span tokens; specials/sentinel/padding are not removed by the evaluator. Zero-padding the resulting probabilities to length256 occurs after softmax and adds no probability mass. Do not sum every token as a grounding score.

Both modes rank scores descending. Official Top1/5/10 accuracy is whether any of the first k candidates has IoU **strictly greater than** .25 or .5. The official code only materializes the first ten rankings. All256 best-IoU recall is a new label-assisted coverage diagnostic. For k>1, `mean_best_iou_within_topk` is also label-assisted: it measures best geometry within retrieved candidates, not a deployable selected-box mIoU. Top1 mean IoU remains an actual readout measure. Avoid calling candidate coverage exact physical-instance identity.

The official `torch.argsort(..., descending=True)` is not stable by explicit contract. Native full ranking/top10 indices should be sealed to preserve real device tie behavior. An independent stable query-index sort is only a disclosed diagnostic when native order is unavailable; it cannot silently replace official ties.

The complete `_main_eval_branch` clamps every `pred_size` component to at least1e-6 after forward/loss and before evaluation. Therefore official primary boxes use this clamp, while raw model dimensions are retained separately for diagnostics. The evaluator's negative-size assertion is not the complete inference behavior. Record the number of clamped dimensions/candidates rather than silently changing or dropping them. Nonfinite raw outputs still constitute explicit failures; no target is silently pruned. Unexpected model/input failures are errors, not additional official-span exclusions.

## Last/proposal and the legacy mIoU accumulator

`train_dist_mod.evaluate_one_epoch` evaluates seven prefixes for six decoder layers: last, proposal, and heads0–4. Top-k accuracy counters are keyed by prefix and BBS/BBF. However, BBF's top1 IoU enters shared `dets['iou']`/`dets['num_iou']` without a prefix key. Thus its final printed mIoU mixes all seven prefixes. Reproducing that legacy aggregate is different from reporting final-layer mIoU.

Report last-layer BBF/BBS as named primaries, proposal separately, every retained head as diagnostic, and the seven-head BBF legacy aggregate explicitly. The aggregate is the mean over eligible object×head observations, not seven independent samples. Official logs do not compute BBS mIoU; a BBS mIoU extension must be labelled as such.

## Scope and denominator contracts

The current checkpoint is the official mixed-platform union checkpoint; all three full VAL splits have been used in prior research/official validation. This is descriptive baseline evaluation, not a fresh method confirmation, source-only transfer, or evidence for TTA benefit. No source-only training is resumed.

Primary official-eligible scores use the GT-class-assisted span on their eligible object set. A separate all-raw text-only first-noun cohort reports every forwarded raw target, including targets the official span protocol excludes. For causal readout comparisons, also retain text-only scores restricted to the official-eligible cohort; comparing unequal denominators alone cannot establish a readout gain.

Use official sample-micro counts/means as the primary descriptive benchmark. Also expose per-scene counts and equal-scene means separately. Multiple targets/frames and multiple heads from a scene are not independent scenes. Any uncertainty interval must resample scene clusters within platform and remain descriptive; no pooled cross-platform headline is needed.

The planned B=1 run has the same model and readout formula but is not an exact B=48 numerical replay. BBF normalizes across the actual batch language length, and the original evaluator does not explicitly remove pad positions. Single-item padding/attention and numerical kernels can change outputs relative to a mixed-length batch. Preserve B=1 as an explicit execution condition rather than claiming published-number bitexact replication.

## Independent CPU scorer contract

All new source prediction files, native score rankings, protocol/code/tokenizer/checkpoint identities and raw-enumeration/skip ledgers must be sealed and hash-verified before the independent scorer opens target classes/boxes. Any dataset-container access in preparation is recorded accurately, not described as never opening a label-containing file. The scorer is a separate process. It uses a three-dimensional convex intersection implementation independent of the official horizontal-polygon-plus-height geometry. Original target metadata, per-candidate GT-IoU vectors and detailed association records stay private; public artifacts contain code/protocols and explicit scalar/count/hashed-scene result projections only.

## Revision provenance

This v2 corrects the interpretation of two unused raw fields only. The original `OFFICIAL_PROTOCOL_AUDIT.md` remains unchanged with SHA256 `29e409ebcec0387f42ac1196b5c23d08742239aee3a6a9c094f51f6c51161fa7`. The prior schema investigation is recorded in `crossplatform/bbox_schema_audit.json`; it supports uncertainty about those extra-field meanings and confirms that the official dataset uses the first seven values. Its private per-object records are not reproduced here. This document does not promote its speculative alternative field interpretation to an established schema. The scoring code already truncates to seven values; no prediction, target transform, numerical result, method or execution boundary changes. No full-VAL target metadata is opened for this correction.
