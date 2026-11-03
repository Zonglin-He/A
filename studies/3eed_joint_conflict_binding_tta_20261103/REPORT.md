# R55: joint subject/reference contradiction supervision and matching projection TTA

Decision: do not adopt the fixed recipe. Source balanced DEV mIoU is 21.830331%; true joint direct readout is 21.720982% (-0.109349pp). Matching LoRA1/3step marginal readouts retain source Top1 on all336; joint readouts after TTA exactly reproduce direct choices. Real TTA was executed on244 inputs with nonzero observation conflicts, not withheld behind a G gate.

## Fixed scope and actual mechanism

Waymo-only epoch100, original256 candidate boxes frozen, exposed DEV336/168 perplatform/twoDrone+threeQuad scene clusters. R53 same full-sentence point/text/query banks are reused; all roles use this same256-box bank, without NMS/top3 cutoff or merging queries into certified instances. Candidate/reference coverage is not guaranteed. Duplicate query slots are not certified distinct instances; their multiplicity can bias categorical mass. Shared-support exclusion prevents treating an overlapping surface pair as spatial counterevidence, but does not solve that multiplicity or certify semantic identity. No new full grounding/backbone forward, source training, visual teacher, image encoder, EMA, FINAL384 access, full5912 scoring, or independent-observation validation is performed. The whole trained source fingerprint is 0eb178a6de52b529d939fa4ae3bfa7b0459d95b7425c3d8a21f473356f2874a9.

Text roles and literal reference spans are inherited. T preserves the actual source firstnoun readout mask; each R uses its own literal span in the complete sentence. All roles re-evaluate native ALL-token logodds after each update. Equal numeric dimensions are not treated as a pretrained RGB semantic space. Global one-entity position scores and R51 V/P/R summation are not added.

Only the existing matching projection final Linear288->64 receives rank4/alpha4 LoRA: A4x288+B64x4=1408 parameters. Text projection, earlier query MLP layers, decoder, backbone, boxes and all other weights are frozen. Adam lr5e-4, betas.9/.999, epsilon1e-8, weightdecay0, gradientnormcap1, fixed1/3steps; source+optimizer reset per observation/variant. ZeroB initializes exactly at source.

For each reference r, candidate distributions are derived from softmax of native role logodds, temperature1. Their product represents a factorized joint hypothesis pT(i)*pR(r,j), not calibrated instance truth. The direct structured readout marginalizes references: target score log pT(i)+sum_r log(sum_j pR(r,j)*(1-C(r,i,j))). This retains alternative references instead of forcing a single positive or declaring a pairwise winner globally correct.

Loss is mean_r[-log(1-sum_ij pT(i)*pR(r,j)*C(r,i,j))] plus source-relative entity logodds MSE plus.01 mean-square realized projection-weight increment. The observation conflict C stays fixed through all steps. Both subject/reference scores change; R54 fixed role responsibilities are not reused. Multiple references use a conditional-independence star model, not a dense scene graph with shared uniqueness constraints.

## Observation conflicts, availability and controls

Only actual registered rawXYZ, RGB and camera fields are used. R51 all-cloud z-buffer visibility is reused, but old grouping/ownership/W matrices are not read. All2,952,487 visible point/pixel RGB values are independently checked in this round. No projected-rectangle background statistics or nearest-color imputation.

Spatial contradictions use only LEFT_OF/RIGHT_OF with declared camera_image frame and explicit graph endpoints. Existing parsed edges often mean R left/right of T; direction is respected. A query needs all eight projected corners positive-depth and inside image plus>=8 visible points. Its conservative x interval contains both projected box-center and visible10/90percentile positions, expanded5percent projected width. Opposite intervals must be separated by.02image width and share<=.1of the smaller visible-point support. Visible surfaces are not declared complete object boundaries or accurate object centers. Camera fields/projection are inherited common dependencies, not physical calibration ground truth.

Color uses three sourcebox support ranges(.85,1,1.15); each needs>=16 pixels and visible spans>=.35of projected box width and height. Exactly one recognized nonnegated color is required. A stable distant hue/black-white alternative must have descriptor>=.45, requested color<=.20, and margin>=.30 on all three supports. Confidence is capped.95. Complex attributes, adjacent hue ambiguity, missing/truncated support and unmeasurable properties remain unavailable; missing expected color alone is not a conflict. These thresholds are prespecified engineering hypotheses, not guarantees of foreground purity or correctness.

C combines geo/target-color/reference-color as1-(1-Cgeo)*(1-CT)*(1-CR), cap.95. Missing reference evidence never creates a contradiction. All256 hypotheses, including unmeasured ones, remain possible. No learned UNKNOWN class is added. Before first TTA, a clarification fixed each entity's source probability mass in conflict-participating/nonparticipating candidate strata; only within-stratum redistribution is trainable. These strata are defined by any observed row/column conflict, not true observability/background labels. This stops increasing unmeasured-stratum mass, but may constrain valid correction and does not certify its source mass.

RGBshuffle permutes actual visible RGB values, so measured color availability can change. Endpointshuffle permutes only the reference columns of geometric conflicts, while native scores and color evidence remain fixed. Each control uses its own fixed conflict strata. Thus these are diagnostic signal controls, not equal-coverage causal certifications of identity. No observations or losses are tuned after task scoring.

## Overall results

| Arm | Drone mIoU % | Quad mIoU % | Balanced mIoU % | Delta source pp | Acc25 % | Acc50 % | Improve/worsen | Query changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0/0 | 0 |
| zero_step | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0/0 | 0 |
| real_direct | 15.491376 | 27.950588 | 21.720982 | -0.109349 | 40.476190 | 21.130952 | 1/3 | 11 |
| real_tta1_raw | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0/0 | 0 |
| real_tta1_joint | 15.491376 | 27.950588 | 21.720982 | -0.109349 | 40.476190 | 21.130952 | 1/3 | 11 |
| real_tta3_raw | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0/0 | 0 |
| real_tta3_joint | 15.491376 | 27.950588 | 21.720982 | -0.109349 | 40.476190 | 21.130952 | 1/3 | 11 |
| RGBshuffle_direct | 15.053778 | 28.123953 | 21.588866 | -0.241465 | 40.178571 | 21.130952 | 1/5 | 13 |
| RGBshuffle_tta1_raw | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0/0 | 0 |
| RGBshuffle_tta1_joint | 15.053778 | 28.123953 | 21.588866 | -0.241465 | 40.178571 | 21.130952 | 1/5 | 13 |
| RGBshuffle_tta3_raw | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0/0 | 0 |
| RGBshuffle_tta3_joint | 15.053778 | 28.123953 | 21.588866 | -0.241465 | 40.178571 | 21.130952 | 1/5 | 13 |
| endpointshuffle_direct | 15.330312 | 27.950588 | 21.640450 | -0.189880 | 40.178571 | 21.130952 | 2/4 | 15 |
| endpointshuffle_tta1_raw | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0/0 | 0 |
| endpointshuffle_tta1_joint | 15.330312 | 27.950588 | 21.640450 | -0.189880 | 40.178571 | 21.130952 | 2/4 | 15 |
| endpointshuffle_tta3_raw | 15.539892 | 28.120770 | 21.830331 | +0.000000 | 40.773810 | 21.130952 | 0/0 | 0 |
| endpointshuffle_tta3_joint | 15.330312 | 27.950588 | 21.640450 | -0.189880 | 40.178571 | 21.130952 | 2/4 | 15 |

Direct true evidence changes11queries:1IoU improvement,3deteriorations and7zero-IoU changes. Acc25 falls40.773810->40.476190%;Acc50 remains21.130952%. Matching1/3step marginal choices exactly equal source, while their conditioned choices exactly equal direct joint readout for every variant. This is absence of extra task value, not absence of real score/parameter changes.

## Optimization and entity movement

| Variant | Active/nonzero updates | Total objective lower | Conflict term lower | Reference Top1 changes | Median target distribution L1 | Median reference distribution L1 |
|---|---:|---:|---:|---:|---:|---:|
| real | 244/244 | 170 | 214 | 10 | 0.00024698 | 0.00343792 |
| RGBshuffle | 212/212 | 150 | 188 | 9 | 0.00014461 | 0.00382787 |
| endpointshuffle | 244/244 | 171 | 223 | 3 | 0.00027185 | 0.00258621 |

On real evidence,170/244 lower total objective and214/244 lower the conflict term. Reference distributions move more than target distributions, with10 reference argmax changes but zero final target changes in raw marginal readout. This is compatible with reducing conflict by rebinding references; reference GT is unavailable, so these changes are not certified as correct/incorrect. Median target/ref distribution L1 is.000247/.003438. These post-score descriptive quantities cannot isolate normalization, source keeping, optimizer budget and evidence specificity as unique causes.

## Does conflict reach recoverable source errors?

All256 candidate IoUs were computed only after predictions were sealed, in the same one-pass task scoring process. They are offline localization proxies, never inputs to C, normalization, updates, stopping or readout. They do not identify semantic instance truth or reference identities.

| Platform | Source wrong25 but any candidate25 | Original Top1 has any conflict | Best-IoU box more penalized than source | Source good25 | Good source has any conflict | No candidate25 |
|---|---:|---:|---:|---:|---:|---:|
| drone | 73 | 29 | 10 | 53 | 25 | 42 |
| quad | 58 | 33 | 8 | 84 | 49 | 26 |
| balanced_all | 131 | 62 | 18 | 137 | 74 | 68 |

Of131 source-failed yet candidate-recoverable observations,62 have nonzero conflict against sourceTop1, while18 assign more conflict to the best-IoU candidate than to sourceTop1. Of137 sourcegood25 observations,74 also receive nonzero conflict. Any positive weight can be small, and another valid reference may explain it; these counts are not an error rate of certified joint bindings. They show neither broad selective correction nor an identity guarantee.68observations have no candidate exceeding.25, limiting selection-only recovery.

## Scene boundary

| Scene | n | Source | Real direct | Real TTA3 marginal | Real TTA3 joint |
|---|---:|---:|---:|---:|---:|
| D_S01 | 106 | 10.584991 | 10.584991 | 10.584991 | 10.584991 |
| D_S02 | 62 | 24.011174 | 23.879711 | 24.011174 | 23.879711 |
| Q_S01 | 62 | 10.117522 | 9.656385 | 10.117522 | 9.656385 |
| Q_S02 | 49 | 32.998900 | 32.998900 | 32.998900 | 32.998900 |
| Q_S03 | 57 | 43.509770 | 43.509770 | 43.509770 | 43.509770 |

Two of five exposed scene clusters decline under true direct readout, three unchanged; none improves on scene mean. These are reused developmental scenes, not an independent confirmation cohort. No frame-independent confidence interval is claimed.

## Audits, execution and retained failures

Predictions and observation artifacts sealed before the single postseal task score. Independent audit checks all2016 fitted endpoints, the native projection against trained checkpoint, all17readouts on336, source/zero exact identity, original boxes, all RGB bytes, joint arithmetic and fixed-stratum mass. Maximum score error8.21e-6 under prespecified5e-4 FP64/CPU versus float32/GPU tolerance; all Top1 agree. Unknown-stratum mass error<=1.23e-15. Separate observation audit independently checks464384 color-entity-query cells and37421056 directed relation cells;4 hash-selected rawcases×3scales×256 boxes reconstruct descriptor values exactly.

Independent gradient audit on4 input-hash-selected active cases(2/platform) reconstructs12 gradients, max norm error1.87e-8. Zero first-step A gradient with Bzero is expected. Some controls are identical where only one evidence type is available; zero gradient differences are retained. Frozen-scope/version assertions and final source fingerprint pass.

Task table retains 5712 values (5040new task-GT cells+672exact source/zero reuse), all492directed negative comparisons, UNKNOWN/no-update observations and all5clusters. Additional86016 candidate-IoU cells are private postseal diagnostic values, not additional performance arms. Independent saved-scalar aggregation maxerror7.11e-15pp.

Observation preparation 54.341s; main resumed multi-arm fitting 48.889s plus5.983s first-two-case engineering process.2100 completed optimizersteps plus1failed engineering step; peak main GPU allocation606147584bytes. These are cached-feature multi-arm costs, not deployment end-to-end timings; R53 feature extraction remains upstream cost.

One technical failure is retained: parametrization registered after model.eval left the new module training=True; evaluation-state assertion failed after first step, with no completed case outputs. No stochastic layer was involved. Attempt002 adds model.eval in reset, preserving source/objective/optimizer. Attempt001 program, plan and failure remain. Generic RoBERTa pooler warning precedes strict complete source checkpoint loading; trained model fingerprint and source score bridge pass.

## Disposition

Stop this fixed contradiction recipe, rather than calling reduced conflict loss a useful TTA result. Do not adopt direct joint readout or matching LoRA1/3steps, do not access FINAL384, and do not start full5912 or another loss/layer/rule sweep. Keep the previous R50 external direct baseline. No best update layer or broader impossibility of joint binding TTA is established.

The next scientific requirement is a more discriminative and independently grounded exclusion signal, including role/reference coverage and support purity. Current joint negatives underdetermine whether target or reference should move. Any future change needs to distinguish supervision quality from optimization/keeping effects; this round does not authorize or schedule one. Do not return to case-specific threshold repair.

## Literature boundaries

CORE-3DVG supports pairwise context modeling but its relation training chooses a target row using GT: [original paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/9b91ee0da3bcd61905fcd89e770168fc-Paper-Conference.pdf),[official model](https://github.com/yangli18/CORE-3DVG/blob/main/models/core3dvg.py). LARC supplies language-regularized structured training, using QA and LLM-derived constraints, not this TTA setting: [original arXiv](https://arxiv.org/html/2404.19696v1). Partial-label learning assumes a true label in its candidate set: [Lv et al.,ICML2020](https://proceedings.mlr.press/v119/lv20a.html). R55 is a task-specific prototype inspired by these mechanisms, not their reproduced baseline, guarantee or established novelty.

Public files contain own code/configs, anonymized task/fit/proxy scalar tables, costs/audits and all negatives. Raw images/pointclouds/captions/coordinates/original IDs/features/weights and third-party source snapshots remain excluded.
