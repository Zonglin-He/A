# R32 independent CPU audit: primary passes, legacy numerical warnings retained

**Primary selected readout, matched QP readout, DEV-only selection, task scores and content disjointness PASS.** The original legacy R30-feedback-memory control is not bitportable across CPU/GPU float32 implementations, and this warning is not suppressed. Original primary predictions, selected families and sealed GPU controls were never modified. This reviewer ran CPU computation only and no optimizer.

## What was independently reconstructed

The audit reconstructed all19 accepted-observation group QPs by an independent active-set projection and checked feasibility, nonnegative dual multipliers, stationarity, complementarity and minimum objective. Each group contributes once to the L2 objective; tripling every query within every group leaves corrections/objective unchanged. There are no constraints on unsupervised candidates and no inferred negative labels.

Candidate support is built from the original accepted positive/negative endpoint groups. For each current candidate, similarity is maximized over the256queries of each A observation, then the nearest k **distinct observations** are selected. Duplicated queries do not become additional observations. The row-support variant marks an entire feedback-containing A row; it remains an ablation and is not pooled into the four primary endpoint key/k families.

The full-caption key uses literal-caption offsets, excludes special and appended not-mentioned tokens, and does not silently reduce the text to a subject category. These keys and all support membership use frozen source features and original runtime pseudo feedback; no GT feedback filtering occurs.

The readout uses **log(p+1e-6)-log(1+1e-6-p)** on saved official float32 probabilities. It is a finite strictly monotone scoring convention preserving official saturation ties, not R30's exact differentiable training log-odds. All528 source/R30 endpoint orders and1600 saved endpoint pairs across the400 historical inputs were verified, including a saturated float32 tie stress case. Binary gate endpoints are explicitly restored to the native source or adapted relative scores rather than relying on cancellation arithmetic.

For QP transfer, libraries store RAW group shifts. After retrieval, the correction is anchored to the current observation's source q0. Translation invariance under candidate-dependent gates and norm/mean global-strength controls were independently checked. A norm match is only a norm match, not the same correction direction.

## Primary numerical and selection results

On all newT128 observations, the entire primary endpoint/noun/k1 score vector and the matched QP-memory score vector match independent CPU reconstruction **exactly: maxerror0**. All new locked readout choices match. Source and R30 head probabilities were separately replayed from saved weights: maximum errors1.20e-6 and9.52e-7, with128/128argmax matches each. The captured source projection is bitexact to the original source projection, the source checkpoint hash is verified, and the actual source input is16384×3XYZ.

The independent DEV procedure reconstructs880 A-feasibility pair decisions and2560 DEV family task scores. All DEV scores, the complete family table and chosen specifications match. Primary endpoint families are exactly the four key/k combinations; row-support and memory/global controls remain their separately declared comparators. R/S/T labels are not inputs to the selector. The selection hash is bound to the T128 prediction seal.

All528 task records and **6272 selected-box task values**, including **1664 values for all13T128 arms**, were reconstructed from saved selected queries, frozen boxes and bound metadata using the authorized independent CPU geometry. All783 negative method/reference comparisons match. The complete GPU-closure receipt verifies256 fullmodel forwards,1664 exact argmax matches, unchanged boxes/nonmatching weights and bitexact source restoration; this auditor verified that receipt but did not execute GPU work.

## The row-k4 boundary warning does not alter selection

Three historical row-support/noun/k4 observations have a CPU/GPU gate fraction difference of1/256. The differing cases include a DEV observation, but **none changes any DEV candidate choice, task score, family ranking or selected primary/comparator**. Endpoint primary reconstruction is unaffected.

The nearest-four boundary candidates have cosine gaps as small as5.96e-8, consistent with float32 ordering sensitivity. Original GPU neighbor indices were not saved, so the exact differing neighbor cannot be certified from aggregate gate fractions. ROW_K4_NUMERICAL_BOUNDARY_DIAGNOSTICS_PRIVATE.json retains the candidate boundary evidence and does not force a new label.

## Legacy R30 memory warning: mainly final-score near ties

Literal legacy normalization was reconstructed separately, including its stacked-array normalization. It is mathematically close to the new noun key but should not be assumed bitidentical. NumPy32 differs from the original GPU legacy feedback-A memory in7Drone queries; literal TorchCPU32 differs in3Drone plus1Quad queries. The all-A-zero-U memory has no such differences in this audit.

For the discrepant cases, the CPU best score exceeds the GPU-chosen candidate by **0 to7.45e-9**. CPU nearest-observation gaps are at least3.01e-4 and the smallest relevant within-A nearest-query gap is9.54e-6. NumPy/TorchCPU nearest observation sets and accepted endpoint-role sets do not cross groups in these cases. This is therefore chiefly a final weighted-addition/ranking near-tie issue, not evidence of neighbor identity switching. Actual GPU neighbor indices are unavailable, so GPU-role switching is not asserted either way.

Query changes can change IoU. For example, originalGPU T_drone_030 query93 hasIoU.04194045 while the literal CPU query148 hasIoU.05529907; T_drone_019 originalquery90 hasIoU0 while TorchCPU query98 hasIoU.04929000. All per-case query/IoU and neighbor-gap evidence is retained privately. These differences must not be presented as all13independent CPU readout argmaxes reproducing exactly. The original GPU fullmodel closure is a separate exact measurement on its own arithmetic path.

A fixed float64 normalization/retrieval/weighted-score path with stable index tie-breaking was specified and sealed as a **posthoc numerical control**, not selected by performance. Its feedback-A memory Drone mIoU is4.736061 versus originalGPU4.358852; Quad remains23.953460. The all-A-zero-U memory is unchanged. This control does not replace the original result or become new confirmation. Its own predictions and all numerical failures remain preserved.

## Content and scientific scope

Actual decoded RGB and canonical unique finite float32XYZ were recomputed for all128T inputs. Both128-content sets are distinct and disjoint from prior656 plus declared history. Current previous-asset byte bindings match the independently decoded inventory. Only the **hash lists** of U128 and FINAL384 were read to verify reserve disjointness; their row manifests, raw media and GT labels were not read.

T128 has one Drone scene and three Quad scenes. The scoped-versus-source effect is confined to a single effect-bearing Drone cluster. A resampling output [0.257055,0.257055] cannot be a population uncertainty interval or significance test. The source/QP/global-amount comparisons correctly mark population CI unavailable; even nondegenerate conditional resampling of Quad cannot estimate unobserved between-Drone-scene variation.

R32 reuses R30's adapted heads with a support-localized readout and cache; it contains no new neural training and is **not a GRACE reproduction**. New exact content does not erase historical fullVAL/scene exposure. The small task gain is a measured result within this protocol, not proof of broad cross-scene superiority.

## Artifacts

AUDIT_RESULT.json is the layered verdict. READOUT_RECONSTRUCTION_RESULT.json, SELECTION_TASK_AUDIT_RESULT.json, CONTENT_AUDIT_RESULT.json and SOURCE_HEAD_AUDIT_RESULT.json separate distinct evidence levels. LEGACY_PRECISION_AUDIT_RESULT.json and legacy_precision_attempt001 preserve the posthoc numerical control. Files markedPRIVATE, the pickle readout cache and predictionNPZs are local private diagnostics and must not be exported as public raw data. The initial auditor reconstruction is retained separately to avoid hiding its failed strict legacy replay check.
