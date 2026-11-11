# R73 — Unified relation spans, explicit image position and one backtracked matching update

**Decision:** keep the parser implementation correction and monotone-step safeguard; do not adopt this absolute-position TTA recipe or expand its performance evaluation. On the fixed 64-input mechanism set, neither the new direct reader nor matching TTA changes any source winner. This is a bounded mechanism test, not evidence that every expert-free or spatial constraint is ineffective.

## Scope and input selection

The attached review text is the execution specification. Its linked sandbox ZIP was not provided as a local attachment; its claimed tests are not treated as executed evidence. R73 independently implements and passes 14 finite-grammar/interval tests. No archive download, source retraining, new visual model, new loss or update-layer search is performed. FINAL data/manifests are untouched; prior source and comparator registrations remain unchanged.

All 2,640 previously exposed descriptions are analyzed for parser coverage only. A separate fixed 64-input mechanism set is selected without task GT: explicit camera/image left/right constraint attached to the first recognized subject, no negation, no earlier competing entity/relation or independent clause, and at least two existing fully projected candidate-group representatives. Select 16 inputs per VAL/TRAIN × Drone/Quad cell by a fixed SHA ordering. Eligibility uses text and candidate projection only, never source correctness, candidate IoU or whether an input will improve. Existing description content after the attached positional clause is not treated as a new global identity label. This is condition-selected, historically exposed development evidence, not a representative all-target score or blind confirmation.

| Cell | Eligible inputs | Selected |
|---|---:|---:|
| VAL1280 / drone | 371 | 16 |
| VAL1280 / quad | 189 | 16 |
| TRAIN1024 / drone | 263 | 16 |
| TRAIN1024 / quad | 87 | 16 |

## Parser correction and finite constraint types

The direction branch of ALLREL now directly embeds REL_RE.pattern. This fixes inconsistent match starts for optional on/to/the/side forms and keeps target-attribute truncation aligned with the relation. The original R72 programs and outputs are unchanged. On the 2,640-description list, 164 first-direction span comparisons recover; 72 literal target-color requests recover after reference colors no longer leak into that fragment. These are syntactic/request counts, not certified correct identity labels or rescued grounding predictions.

The finite grammar classifies 1,051 descriptions as IMAGE_ABSOLUTE, 13 as ENTITY_RELATIVE and 1,576 as UNKNOWN. Original character spans are retained privately. The parser statistic simple_entity_old=0/simple_entity_new=13 compares the original-vs-unified span test *within this new finite ENTITY_RELATIVE subset*; it must not be read as an assertion that R72 globally had zero simple relations (R72 reported five under its different acceptance path). The complete anonymous parser-coverage and UNKNOWN-reason table is exported.

Only the absolute image position becomes new supervision here. The separate parse-only comparator retains the complete R72 numerical support/color/anchor gates and recomputes support descriptors only where the corrected literal requests require it. Entity-relative phrases remain classified for coverage, without adding an unverified anchor detector or removing their source-identity uncertainty. Roads, vehicle-relative sides, previous independent clauses and earlier entity relations are not silently reinterpreted as image halfplanes. This finite grammar is not a general language parser.

## Position evidence and its limits

For each unchanged source A256 box, project all eight full-box corners through the existing calibrated camera convention. Require positive depth and all corners inside the image, with no clipping. Use the fixed existing candidate grouping; a group is known only when all its members have valid full projections. Its rectangle is the union of their full projected boxes. For an explicit image-left request, reject only if the entire interval begins to the right of normalized x=0.55; for image-right reject only if its entire interval ends to the left of x=0.45. The midpoint=0.5 and margin=0.05 are fixed engineering assumptions before scoring. Border crossing, partial visibility and invalid projections remain UNKNOWN.

The new position path does not require color, RGB surface support or a reference object. It uses language and camera geometry, not RGB pixel semantics, and is not called full three-modality self-supervision. The RGB/color path remains the unchanged separate R72 component. Candidate grouping can be imperfect and projected source boxes are not measured object masks. Calibration also inherits the declared zero-distortion assumption when metadata omits coefficients.

## Task results on the fixed mechanism set

| Method | Drone mIoU % | Quad mIoU % | Equal-platform mIoU % | Acc@25 % | Acc@50 % | Query changes / better / worse vs source |
|---|---:|---:|---:|---:|---:|---|
| source | 12.4910 | 26.2939 | 19.3925 | 37.5000 | 17.1875 | 0 / 0 / 0 |
| old_R72 | 12.4910 | 26.2939 | 19.3925 | 37.5000 | 17.1875 | 0 / 0 / 0 |
| parse_fix_direct | 12.4910 | 26.2939 | 19.3925 | 37.5000 | 17.1875 | 0 / 0 / 0 |
| position_direct_opt | 12.4910 | 26.2939 | 19.3925 | 37.5000 | 17.1875 | 0 / 0 / 0 |
| position_TTA_backtrack1 | 12.4910 | 26.2939 | 19.3925 | 37.5000 | 17.1875 | 0 / 0 / 0 |
| position_mismatch_direct_opt | 13.7277 | 24.8710 | 19.2994 | 39.0625 | 17.1875 | 15 / 5 / 4 |
| R59 | 12.9635 | 26.8300 | 19.8968 | 39.0625 | 17.1875 | 10 / 4 / 1 |
| CLIP_reference | 13.5184 | 26.6388 | 20.0786 | 37.5000 | 17.1875 | 31 / 11 / 11 |

The old_R72 row is its geometric direct reader, not a new rerun of its three-step optimizer. Source/R59/frozen CLIP are exact cached reference readers on these same 64 inputs. R59 has extra point-sampling information; CLIP adds external pretrained visual knowledge. They are system comparators, not evidence that the new parameter-update algorithm is stronger or that every no-expert system must fail. CLIP is attached only as a readout reference after the corresponding evidence/fit and is never used in feedback, selection, line search or loss.

The position-mismatch control swaps left/right while preserving the phrase role, candidate pool, camera geometry and fixed evidence-processing budget. It changes 15 queries, improves five IoUs and worsens four, with net -0.093112 pp. This illustrates why endpoint changes or favorable cases alone are not evidence of reliable supervision. The main true-position reader and TTA change zero queries. Per-cell, all scene, duplicate-frame and ordered negative comparisons are retained.

## Decision-relevant coverage

| Diagnostic | Inputs |
|---|---:|
| Nonempty new negative set | 61 |
| Source winner rejected by the true constraint | 0 |
| Source winner with complete legal projection | 39 |
| Source winner with incomplete/invalid projection (UNKNOWN) | 25 |
| Source failures at Acc@25 | 40 |
| Source failures at Acc@25 with a known projected winner | 26 |
| Source failures with an Acc@25-capable candidate elsewhere | 33 |

All 39 source winners whose projection is verifiable already satisfy this coarse horizontal halfplane. Twenty-six nevertheless have target IoU <=0.25, so satisfying the halfplane is insufficient for grounding. This does not certify that all 26 select the wrong physical instance: center/size/yaw errors can also lower IoU. The other 25 winners remain UNKNOWN, not validated correct and not forced negative.

The pre-GT query-vs-group audit finds zero violating source winners even before conservative group propagation, with the same 39 known source winners. Thus the zero decision coverage in this set is not caused by merging an individually rejected source winner into an uncertain group. There are zero rejected queries with target IoU >0.25; that localization-only diagnostic is not semantic identity precision. No true instance masks or separately verified physical-identity labels were obtained.

With negative set N and the same reject+KL objective (lambda=1), the exact no-update distribution optimizer has negative mass P0(N)/2, maintaining within-stratum source proportions. If the original winner is outside N it remains the winner. The direct reader therefore cannot correct any current decision in this mechanism set. Shared matching weights can affect other candidates, but those changes do not acquire additional identity supervision merely because optimization succeeds.

## Single proposal, label-free acceptance and actual parameter changes

Keep the existing 184,960-parameter 288→288→288→64 matching MLP, frozen source features/tokens/boxes, native log-odds-change source bridge and per-input source/optimizer reset. Make one Adam proposal at lr=0.0005 with gradient norm clip=1. Test scales 1,1/2,...,1/256 on the identical fixed negative-set objective -log(1-P(N))+KL(P0||P). Accept the first finite loss <= old loss - max(1e-12,1e-6*abs(old loss)); otherwise restore every weight, Adam moment and counter. No IoU/GT is read by acceptance. The KL target and negative set do not change during trials.

There are 61 proposals, 56 accepted updates and 5 full rejections; three inputs have no proposal. Of the 61 full-size proposals, 60 initially increase the objective. Only one accepted update uses scale 1; the remaining accepted updates need a reduced scale. All accepted loss increases after backtracking: zero. Actual saved weights are independently replayed; maximum independent NumPy probability error is 8.99e-07. On rejection the saved head hash, Adam-state hash and parameter delta verify exact restoration. On acceptance parameter deltas are nonzero. This repairs R72's nonmonotone-step execution issue without demonstrating a task gain.

## Evidence preservation, scoring and costs

Predictions for all eight arms are sealed before lookup of saved R69 per-candidate IoUs. No new raw GT/3D-IoU calculation is performed. The task audit independently rechecks all 512 values, 241 ordered negatives and aggregate/coverage tables. A separate audit reconstructs the pinhole projection algebra for every selected input, validates group rejection intervals, native head outputs, accepted loss drops and rollback hashes. Both audits are different numerical implementations authored by one developer; they are not a separately blinded human identity audit.

The 64 descriptions cover 9 official recording clusters and 63 unique RGB/LiDAR frame pairs. A repeated frame is grouped; no 64-independent-scene confidence interval is reported. Camera/projection evidence preparation takes 6.892s; cached-prefix fitting/readout takes 2.753s. New full 3EED forwards, GPU inference, CLIP encodings and source-training steps are all zero. These are marginal cached-study times, not complete deployment latency; peak RSS was not measured.

The private case ledger preserves every original description, target/constraint character span, UNKNOWN reason, original/new candidate and projection interval, probability vector, objective/trial curve and postseal IoU. No cases are discarded based on outcome. Original descriptions/media/IDs/boxes/coordinates/feature vectors/actual weights stay private; public exports contain anonymous task, coverage, step, cell, scene/frame and negative scalar tables plus owned code and audit hashes.

## Stop decision

Do not expand performance evaluation to full VAL1280/TRAIN1024 because the fixed mechanism test supplies no true-constraint correction. Do not broaden grammar, change margins, add vertical rules or increase update steps after seeing these labels in this round. Keep the shared-regex parser correction as an implementation fix and retain the objective-only backtracking safeguard. This recipe does not establish an additional TTA contribution over the same-information direct reader.

The unresolved mechanism is how to obtain verifiable evidence that distinguishes competing candidates *within* an admissible image region, and how to assess incomplete projections, not how to lower the present auxiliary objective further. These are prospective questions, not new experiments started here. The small condition-selected set does not rule out rare useful absolute-position corrections elsewhere or all expert-free adaptation; it fails the current prerequisite for expansion. No follow-on job, source training, FINAL evaluation or automation is queued.

Publication helper repair: the first local upload-index attempt retained an old common72 import and stopped before any network write. Only that saved-artifact utility import was corrected; predictions, supervision, model fits, task values and all previous audits were retained without rerunning the experiment.
