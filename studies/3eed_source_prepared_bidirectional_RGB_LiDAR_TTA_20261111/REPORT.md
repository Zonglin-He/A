# R76 — Source-prepared geometry-local bidirectional RGB–LiDAR grounding

## Decision

**Retain the frozen bidirectional architecture as a development candidate; do not adopt the current physical-recovery TTA recipe.** This is a bounded, source-prepared experiment, not a no-source-preparation claim.

On historical DEV336, the bidirectional frozen model reaches **22.611596%** balanced mIoU: +0.781265 percentage points versus original source, +0.319220 pp versus source-prepared independent late fusion, and +0.339409 pp versus source-prepared RGB→LiDAR. Turning off the reverse path in the same trained state costs 0.076934 pp; wrong RGB costs 0.227920 pp. These are real functional task effects.

However, R59 already reaches 22.611116% on the same cohort, only 0.000480 pp below the new frozen model. The new architecture has **not** established a meaningful overall accuracy advantage over that strong frozen reference. Its improvement is concentrated on Drone, with little Quad gain and losses in two of five recordings. These results are not a production adoption or novelty claim.

Correct TTA1 and TTA3 both reach 22.611510%, **−0.000086 pp** versus their own frozen model, with no Acc@25/50 improvement. The corrupted-interaction TTA3 control is higher at 22.669000%. Therefore the adaptation increment and its claimed dependence on correct physical correspondence are not established. No source epoch, learning rate, mask fraction or update scope was changed after observing target scores. No extra budget or full-target expansion is queued.

## Authorized protocol change and scope

The user now authorizes paired Waymo RGB/LiDAR/language and source grounding labels to prepare new modules. Original Waymo-only epoch100 checkpoint, point backbone, RoBERTa, cross encoder, decoder, proposal/regression weights remain frozen. A **copy** of the existing matching projection is source-trained within the new module; the active prepared model therefore has a changed matching function even though the original checkpoint file and frozen base fingerprint are unchanged. This is source-prepared multimodal TTA, not an assertion that no source-supervised parameters changed.

One frozen 3EED is shared with a small randomly initialized RGB encoder and interaction module. No ImageNet/CLIP/VLM/SAM/detector model or its cached predictions/features are used. The source-aligned R25 records provide corrected original 3EED boxes and saved **source GT candidate IoUs**, not pretrained RGB representations. No new pretrained visual network or second complete 3D grounder is added. RGB reads raw paired images.

The final experiment fixes 512 Waymo TRAIN descriptions and 128 Waymo VAL descriptions by recording-round-robin and hash, without GT/visibility/target filtering. They span 94 and 82 disjoint recording groups, with no common point file. Three models use identical seed0 and the same eight epochs, optimizer and sample orders. Waymo VAL chooses only the best **trained epoch 1–8**, earliest on ties. Original epoch0 is a baseline, not a fallback candidate that could mask failed training. No target labels or target feature tensors enter source training/selection.

Target evaluation is the already exposed DEV336, 168 descriptions per platform, five recording groups and 336 point/image frame pairs. These are development results, neither a new blind test nor ALL5912. FINAL is not accessed. Source preparation does not turn historically exposed target data into independent confirmation.

## Architecture: genuine local bidirectional interaction

RGB: three stride-two 3×3 convolution blocks, GroupNorm and GELU, channels 3→16→32→64. Images are resized to 384×256 with exactly corresponding projection scaling, yielding 48×32=1,536 RGB tokens. No pretrained visual initialization. LiDAR: 1,024 original 288-D frozen pre-language seed features and spatial positions. Each branch has a small residual bottleneck adapter.

Calibrated local graph, built before grounding labels/candidate selection: source five-camera view rotation and target Drone z shift follow the verified model/camera convention. A 4-pixel sampled-point depth buffer checks seed visibility, with tolerance 0.3+0.02×camera depth. Each visible point can read only a local 3×3 image-token neighborhood, weighted by projected distance. Each RGB token can read at most eight visible projected points within 1.5 feature cells and a front-depth band 0.5+0.03×front depth. Unknown support has zero exchange. These are physical correspondence reliability constraints, **not certified object masks or instance-identity probabilities**. The buffer uses the actual 16,384-point source sample; missing surfaces/calibration error can still cause association mistakes.

The order is sequential: **LiDAR→RGB updates RGB tokens, then RGB→LiDAR reads those updated tokens**. Thus the reverse branch can influence the 3D representation and final selection; it is not a parallel unused auxiliary output. Local learned Q/K/V attention selects within the admissible sparse graph. Learned scalar tanh residual gates start at zero; their sign is not forced positive. The trained reverse gate is negative, a learned subtraction of a local feature contribution rather than evidence that the reverse branch is inactive.

All arms retain an independent late RGB readout for a comparable appearance path. Without early interaction, the two branches can still combine at candidate matching. This is the requested independently encoded/late-fused control, not an RGB branch that is disconnected from the task. Correct/uni/bidirectional models have the same module schema and initialization, but are independently source-trained and source-selected. Their whole-model differences are method comparisons, not isolated intervention estimates. The same-weight reverse-off and wrong-RGB tests provide narrower interventions.

The 3D tokens remain the spatial carrier and enter the original text cross encoder and six decoder layers. Original A256 query seed indices, six source geometry-position arrays and final boxes are fixed. Features and final query/text scores are recomputed. This prototype adapts **selection among fixed source boxes**, not proposal coverage, box shape or yaw.

## Source objectives and target-copy prevention

Grounding uses the original source GT IoUs for the unchanged A256: soft target proportional to exp(IoU/0.1), candidate cross entropy. If all candidate IoUs are zero, only that grounding term is zero; auxiliary training and the sample remain. Text stays complete, with the inherited first-noun matching mask and original `not mentioned` alternatives. This is candidate-ranking supervision, not full Hungarian detection retraining or GT text-span parsing.

Joint loss: grounding +0.1 point-feature recovery +0.1 masked-RGB recovery +0.01 image-only sparse log-depth regression. Auxiliary heads are separate from the grounding matching projection.

Two complementary geometric mask sets withhold 25% of depth-supported RGB cells and a disjoint 25% of visible non-anchor point cells. RGB pixels are masked **before the CNN**, not merely the last feature. Point values are zeroed before interaction and blocked as reverse keys. All source query anchors remain. Point targets are the normalized first64 channels of the fixed original LiDAR representation, stopped from gradient; RGB targets are measured raw patch means. Geometry says which observations correspond, not which object the sentence names. The predicted depth head reads **preinteraction, image-only features** and never reads LiDAR depth/point features through the reverse branch.

Independent trained-model tests replace all masked point values or masked RGB pixels with extreme values and obtain exact unchanged auxiliary features; depth prediction is invariant to changing all point features. This prevents direct target-value copying along those tested paths. Retained neighboring features/backbone receptive fields still provide context; it is not a claim of reconstructing a genuinely unobserved object, and XYZ does not uniquely determine RGB.

Source optimizer: AdamW lr3e−4, decay1e−4, batch1, clip1, eight fixed epochs. BF16 feature/transformer computation, FP32 matching logits and FP32 source validation. Total trainable prepared module 338,343 parameters, including the source-trained matching copy. Original 3EED parameters are frozen throughout.

## Source preparation result

| Source configuration | Selected trained epoch | Source VAL128 mIoU % |
|---|---:|---:|
| Original source | fixed epoch100 | 45.758121 |
| independent | 6 | 47.148511 |
| rgb_to_point | 6 | 47.454144 |
| bidirectional | 6 | 47.385454 |

The source candidate-oracle mIoU is 56.836274%. All three source-selected checkpoints are epoch6, chosen before target inference. The active bidirectional source improves source validation over original by 1.627332 pp, but does not outperform the unidirectional model there. Source preparation is functional; this does not guarantee target or TTA gains.

Source gain includes new matching/adapter supervision and RGB learning. There is no separately trained matching-only/no-RGB arm in this bounded matrix, so the entire source→new-model increment cannot be attributed exclusively to RGB. Same-weight modality and reverse controls test narrower effects.

## Unlabeled TTA and strong zero-update comparison

At each target input restore the source-selected bidirectional state. Freeze RGB convolutional encoder, original 3EED, prepared matching head, late RGB projection and all auxiliary heads. Update only image/point adapters, the local interaction Q/K/V/output projections and interaction gates: **102,498 parameters**. No EMA teacher or external pseudo-label.

Use the same source-prepared auxiliary loss averaged over two fixed complementary corruptions. Preserve the source-prepared complete-input candidate distribution by KL plus 0.1 normalized-query cosine preservation. These stopped predictions constrain drift; they do not certify target identity. Adam lr1e−4, clip1, fixed one/three-step endpoints; one proposal per step, transactional backtracking over scales1…1/256 using only the identical unlabeled objective. All branches/samples remain, including the rejected proposal.

A correct run and independently reset wrong-interaction-binding run use the same images, masks, optimizer and steps. The latter permutes logical point/pixel interaction bindings and point-feature image readout. Raw RGB reconstruction and observed depth targets stay intact; it is a partial physical-binding corruption control, not destruction of every auxiliary signal.

The frozen three-reader control directly averages complete input and the same two corruption readouts, no weight update. It has the same physical observations and generated corruptions. It is not an equal-wall-clock baseline: TTA uses gradients and repeated trial objectives, with all costs counted below. All endpoints still choose original A256. An alternative RGB image is deterministically permuted within each target platform for the wrong-RGB frozen control; different frames can still share semantics, so it is not a certified unrelated-object adversary.

## DEV336 target results

| Method | Drone mIoU % | Quad mIoU % | Equal-platform mIoU % | Acc@25 % | Acc@50 % |
|---|---:|---:|---:|---:|---:|
| Original 3EED source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| Source-prepared independent branches + late fusion | 16.282575 | 28.302178 | 22.292376 | 39.880952 | 20.535714 |
| Source-prepared RGB→LiDAR | 16.329964 | 28.214411 | 22.272187 | 40.178571 | 21.130952 |
| Source-prepared LiDAR→RGB→LiDAR, frozen | 17.083330 | 28.139863 | 22.611596 | 41.071429 | 21.428571 |
| Same bidirectional state, reverse path off | 16.700915 | 28.368410 | 22.534662 | 40.773810 | 21.428571 |
| Same state, within-platform wrong RGB | 16.799218 | 27.968135 | 22.383677 | 41.369048 | 21.130952 |
| Frozen complete + same two corruption readers | 16.796001 | 28.415060 | 22.605530 | 41.071429 | 21.428571 |
| Bidirectional + physical TTA 1 step | 17.083330 | 28.139691 | 22.611510 | 41.071429 | 21.428571 |
| Bidirectional + physical TTA 3 steps | 17.083330 | 28.139691 | 22.611510 | 41.071429 | 21.428571 |
| TTA 3 steps, corrupted interaction binding | 17.198137 | 28.139863 | 22.669000 | 41.369048 | 21.428571 |
| Historical frozen ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 |
| Historical R59 frozen representation fusion | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |

## What the comparisons establish

Frozen bidirectional improves 44 IoUs and harms 54 versus original source, with 182 changed query selections. The net metric is positive because improvements have larger magnitudes; a larger count of harmed cases is not used as an automatic veto. Acc@25 rescues18 and breaks17, only one net threshold hit. Largest single positive contributes0.229118 pp; the remaining335 contribute0.552148 pp, so this mean gain is not explained by one exceptional case.

Drone frozen gain versus source is +1.543438 pp; Quad gain is +0.019093 pp. Versus R59, Drone is better by about0.697762 pp while Quad is worse by about0.696801 pp: these cancel in the platform-average metric. Two recording means improve strongly, one modestly, two decline; no claim that every platform/scene improves is made and no such gate selects a method.

The same-weight reverse-off comparison loses0.076934 pp overall, but direction differs: disabling reverse hurts Drone and helps Quad. Wrong RGB loses0.227920 pp overall; it still sometimes helps, and its Acc@25 is slightly higher. This supports measured mean appearance/reverse-path contributions in the current frozen system, not perfect association reliability or universal modality dominance.

Correct TTA1 changes two queries and TTA3 three relative to frozen bidirectional. Both have zero IoU rescues and one extremely small harm; Acc@25/50 remain unchanged. The three-step task delta is −0.000085797 pp. It is +0.005980 pp above the slightly weaker frozen three-corruption ensemble but **does not beat its own stronger frozen single-input model**. Do not select the weaker zero-update control and declare adaptation successful.

The wrong-interaction TTA3 control gives +0.057404 pp above frozen from one Drone IoU improvement and no losses. Correct auxiliary recovery therefore has not shown a task advantage over this corruption control. This control is not an eligible final method selected using target GT.

The optimizer is working: 2,015/2,016 proposals accepted, zero accepted objective increases. Correct mean auxiliary loss decreases0.03486450→0.03414626 by step3, while total objective decreases0.03486450→0.03416991. All1,008 saved endpoint adapter states are replayed exactly. The observed limitation is that these continuous changes almost never change useful target decisions at this budget, not zero gradients or stale cached readout. Current evidence does not separate insufficient step magnitude from a weak task direction; the corruption control and prior results do not justify blindly increasing steps.

Postseal-only candidate diagnostic:268/336 inputs have an A256 candidate above IoU0.25;131 original-source failures and130 bidirectional failures still have such a candidate. Candidate coverage is a limit, but does not by itself explain all remaining matching failures. IoU differences are localization proxies, not independently certified object-identity labels.

## Evidence, engineering repairs and resource accounting

All336×12=4,032 target readouts are sealed before one saved candidate-IoU join. No new raw target GT/geometry scoring. All3,534 ordered negatives, fallbacks, unchanged selections and rejected steps are exported. Source validation uses source labels by authorization; target labels never enter checkpoint selection, correspondence construction, auxiliary targets or acceptance. Independent scalar aggregation covers all methods, splits, platforms and updates. Recording/frame tables retain clustering; no i.i.d. CI or unseen-scene claim is made.

Fresh640 source prefixes reproduce the original corrected R25 boxes/query/text/hidden features bit-exactly. Independent camera algebra and sparse graph checks cover all976 prepared inputs (640 source +336 target), maximum projection discrepancy1.908e−6 feature cells. All24 active source validation curves are checked, selected three states replayed on all128 source validation inputs, zero probability error. All target frozen readouts, controls,1,008 adapted states, objectives and final boxes replay exactly; original base fingerprint remains unchanged. The zero-residual initial model reproduces original query/hidden outputs exactly.

Early geometry engineering attempts001–003 are retained: NumPy-container handling, an unsuccessful raw-column guess, and a None target suffix conversion were fixed before any source fitting used those graphs. Final geometry attempt004 passes all five source views and both targets. Independent audit Decimal-to-float conversion was repaired without changing produced graphs.

Source attempt001 was interrupted before any target inference/score after finding a real boundary error: the late RGB linear bias reached candidate anchors with no physical RGB support. The fix masks the **projected residual**, including bias, by support. An independent all-unknown test verifies exact zero RGB delta. Thirteen completed source epochs/partial states and job logs remain, invalid for final model selection. Active attempt002 repeats the same data/seed/optimizer/eight-epoch budget; no target-label tuning occurred. This is an implementation repair, not an additional algorithm winner selected from two attempts.

Active source training/validation:15,360 transformer suffix evaluations,382.07 seconds, peak allocated GPU about1.014GB. Fresh source prefix capture:640 full source forwards/72.59s, plus five engineering forwards. Target production:185.36s. The producer counter3,360 counts explicit task readers only; exact `DERIVED_COST_AUDIT.json` additionally counts6,176 objective forwards, for **9,536 actual target grounding suffix forwards** and13,024 auxiliary fusion-only evaluations. This accounting correction does not alter any weight or prediction. Independent target replay adds5,376 suffix forwards/54.91s; source-selected replay adds384. The interrupted source attempt has thirteen completed epochs (at least8,320 train/validation suffix calls), plus an unmeasured partial epoch; its cost is not silently included as zero.

These are cached-prefix timings, not end-to-end deployment latency. Keeping original A256 anchors/positions requires an unfused source proposal pass before the fused rerun when no cache exists. The model is one shared 3EED reused, not two separately pretrained grounders, but it is **not** a free single-pass plug-in. Source data loading, original target backbone capture, audits and publication are separate costs. Performance/resource comparisons with R59 need a future uncached timing study before claiming an efficiency advantage.

## Relation to prior work and boundary

PiMAE uses projected complementary masks, a two-branch autoencoder, shared decoder and cross-modal reconstruction. R76 borrows correspondence-conditioned interaction and complementary recovery; it does not reproduce PiMAE's full pretraining framework or benchmarks. [PiMAE official repository](https://github.com/antonioo-c/PiMAE).

SLidR uses superpixel-driven point/image correspondence and distillation from pretrained image representations. R76 borrows the geometric correspondence carrier, **not** its pretrained image teacher or superpixel/instance certification. [SLidR official repository](https://github.com/valeoai/SLidR).

xMUDA separates task and mutual-mimicking heads to reduce harmful modal imitation. R76 has independent auxiliary heads and grounding supervision, but no dual semantic-segmentation classifiers, so it is not an xMUDA/MM-TTA/CoMAC reproduction. No class-prototype queue, EMA or externally supplied semantic labels are introduced. [xMUDA official model](https://github.com/valeoai/xmuda/blob/master/xmuda/models/xmuda_arch.py).

**Adoption boundary:** prepared bidirectional fusion shows a useful development mean gain over source and trained single-direction/late-fusion controls; it nearly ties the existing R59 average, with platform tradeoffs. Retain its code and selected weights for research, not production adoption or a settled new method. Current physical-recovery TTA has no useful increment over the stronger same-system frozen model and is not adopted. No automatic source-budget increase, target threshold/step search, new loss or FINAL evaluation follows this result.
