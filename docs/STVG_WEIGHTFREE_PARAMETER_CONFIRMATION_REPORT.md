# Eight-candidate spatial OPD: parameter diagnosis and larger confirmation

2026-10-15. This finite study extends the eight-source development experiment. It does not promote CURRENT or launch full Table 1. All source checkpoints remain the original cross-domain TA-STVG checkpoints; Temporal is Caption–Query Alignment with 512-parameter Adam, Spatial is episodic 1792-parameter Gaussian IoU-OPD with reference-agreement rejection. Only candidate generation is weight-free; R1 and Student are learned models.

## Actual parameter search

Four historical sources/domain are development, explicitly using post-seal GT metrics to choose hyperparameters. Two real LR alternatives, six one-factor alternatives (sigma 0.5/2, tau 0.5/2, steps 3/1), then two development-guided combinations/domain were run. Original eight-candidate outputs are exact saved reuse. This totals 80 new complete development fits and full redecodes, not endpoint parameter interpolation. R1 choices, references, noise keys, temporal predictions and supervision positions are matched. Eleven recipes/domain were considered, with adaptive combinations disclosed. Combination meanings differ by domain: HC combo1=LR0.5×/3 steps, combo2=LR0.5×/sigma2×; Vid combo1=LR0.25×/1 step, combo2=LR0.25×/sigma0.5×.

| Development T+S recipe | HC vIoU % | Vid vIoU % |
|---|---:|---:|
| original | 33.660 | 19.203 |
| lr025 | 33.404 | 22.445 |
| lr050 | 34.234 | 22.097 |
| sigma050 | 33.666 | 21.820 |
| sigma200 | 33.695 | 17.909 |
| tau050 | 33.653 | 19.161 |
| tau200 | 33.664 | 19.284 |
| steps3 | 34.573 | 20.139 |
| steps1 | 33.412 | 22.059 |
| combo1 | 33.732 | 22.234 |
| combo2 | 34.247 | 22.577 |

Sampling sigma and step count matter; temperature changes are smaller on these four sources/domain. This does not establish that dataset heterogeneity is only a parameter issue. Domain counts are equal. All old selected references have five-of-five presence, so raising the visibility threshold 0.6 to 0.8/1.0 changes zero choices and is reported as a zero-effect metadata check. Candidate area bins, DIS cycle threshold, anchor budget, sample count32, Adam betas and carry were not additionally swept. This is a bounded search, not exhaustive optimization.

## Frozen configuration and confirmation design

| Domain | Selected recipe | LR | sigma | tau | steps |
|---|---|---:|---:|---:|---:|
| hc2 | steps3 | 0.01 | 0.025 | 0.05 | 3 |
| vidstg | combo2 | 0.0075 | 0.05 | 0.25 | 5 |

Selection maximizes old-development mean T+S vIoU; ties within1e-12 use smaller LR, then fewer steps, then name. The complete configuration was locked before confirmation Student execution and label exposure. No reselection on confirmation. The new cohort is 16 unique parents/domain, one query/parent, selected by the prior registry's parent-hash order before development tuning results. It excludes the old development/current four parents. These32 sources have historical project exposure, so they are **not an untouched official test set**. They are new to this route's selection-confirmation split. Old-eight and new32 alias names refer to different sources and must not be joined by alias alone.

Inference does not read GT: frozen source/video hashes bind input; Temporal uses its original four candidates; four QA-centered five-frame windows generate at most eight SelectiveSearch/GrabCut/DIS regions plus Source. R1 selects IDs using the unchanged 448px contact-video protocol. Fewer than four eligible source-grid centers with physical±2 coverage produces explicit no-spatial-update fallback. Invalid/abstained/Source choices or insufficient presence provide no new reference. There is no GT candidate supplementation or Oracle fitting. All32 full Student outputs are sealed before official GT scoring.

## Confirmation full-tube results

Query-macro equals parent-macro here because each parent has one query. F=Frozen, T=Temporal-only; original/selected denote spatial-only and T_ prefixes denote the full dual branch. Original uses default five-step eight-candidate OPD, not YOLO/SAM2. No new native YOLO/SAM2 comparison was run on these32 sources.

| Arm | HC vIoU % | Vid vIoU % |
|---|---:|---:|
| F | 32.587 | 9.701 |
| T | 30.565 | 9.466 |
| original | 32.201 | 9.500 |
| T_original | 30.349 | 9.398 |
| selected | 32.640 | 10.144 |
| T_selected | 30.744 | 9.951 |

| Domain | Paired comparison | Mean Δpp | Parent-bootstrap 95% CI | Negative sources | Worst Δpp |
|---|---|---:|---:|---:|---:|
| hc2 | T_selected − T | +0.180 | [-0.478, +0.841] | 2/16 | -3.067 |
| hc2 | T_original − T | -0.216 | [-1.085, +0.588] | 3/16 | -4.175 |
| hc2 | T_selected − T_original | +0.396 | [-0.044, +0.888] | 2/16 | -1.427 |
| hc2 | selected − F | +0.053 | [-0.860, +0.975] | 3/16 | -4.034 |
| vidstg | T_selected − T | +0.485 | [-0.136, +1.423] | 2/16 | -0.941 |
| vidstg | T_original − T | -0.068 | [-1.051, +0.751] | 3/16 | -5.781 |
| vidstg | T_selected − T_original | +0.553 | [-0.096, +1.390] | 2/16 | -1.463 |
| vidstg | selected − F | +0.442 | [-0.167, +1.369] | 2/16 | -1.097 |

The intervals use10,000 paired resamples within domain. All individual positive, negative, neutral and undefined values, tIoU, dense/supervised/unsupervised spatial metrics and official/independent checks are included in JSON. Undefined support remains null. Confirmation partitions use the same prelocked anchor positions for all six arms; historical development F/T and external-expert support partitions are not matched and should only be compared on overall metrics. This limited confirmation cannot substitute for full official Table1 evaluation or prove equivalence to external instance tracks.

## Interpretation of the held-selection confirmation

The selected recipes improve T+S over the default eight-candidate recipe by0.396pp HC and0.553pp Vid; versus Temporal-only the spatial increment is0.180pp and0.485pp. Both paired95% intervals cross zero. This supports a bounded parameter-control benefit but not a statistically established stable spatial improvement or equivalence to YOLO/SAM2.

Temporal-only itself is2.023pp below Frozen in HC and0.236pp below Frozen in Vid on these additional sources. The final selected dual branch is1.843pp below Frozen in HC and0.249pp above Frozen in Vid. Two HC sources have particularly large Temporal-only declines (HC06−30.571pp, HC14−31.293pp). These are confirmation aliases, not the old development aliases. They remain in the mean and are not used to change Temporal or select another spatial configuration. The larger cohort therefore exposes a separate generalization issue that old-eight development gains did not establish away.

The selected spatial increment versus Temporal has two negative sources/domain; worst−3.067pp HC and−0.941pp Vid, versus default worst−4.175pp and−5.781pp. Many sources are exactly unchanged because no new reference is available, the IoU samples are uninformative, or the proposed update is rejected. The execution/decision files distinguish these cases. A zero metric delta can also arise from zero overlap despite changed boxes; it is not automatically a no-update record. Smaller damage is valuable but is not the same as extracting correct identity from every candidate pool. The current finite result is retained without a production promotion or confirmation-driven search.

## Cost and execution

On the original same eight sources, prior native full-video tracking took2921.429s versus65.162s for cached sparse-window eight-proposal computation: a **44.834× stage-time ratio**,97.770% reduction. Work coverage and cache scope differ; this is not end-to-end acceleration. Native tracking covers full videos; the new stage covers four five-frame windows and excludes RGB export/load. Frozen R1, Temporal, Student, loading and scoring remain. Do not extrapolate this ratio to13,785 queries.

Confirmation: 224 Temporal R1 calls, 128 spatial windows, 128 spatial R1 calls, 46 actual Student fits/full redecodes. Temporal worker 154.406s plus separately measured model load5.637s; candidate worker 260.937s with 233.807s proposal computation; spatial R1 generation 56.313s and request total60.942s; Student worker224.135s. These are scoped component measurements, not a single deployment timing. Explicit no-reference/no-temporal-input fallbacks and reused controls are included, not counted as new fits.

## Verification and limitations

Full source recapture verifies checkpoint state, pixels, frame grid, native boxes/logits; actual adapted full-model outputs equal cached decoder outputs, and source state is restored. Independent CPU readback checks reference binding, float64 IoU, saved Adam/parameter rollback and final output geometry. An additional audit reconstructs Gaussian noise, sample coordinates, IoU rewards, softmax weights, weighted losses and Adam updates from saved gradients in float64; it does not independently rerun backward gradients. All confirmation canonical videos are hash-checked, selected pixels freshly decoded, mask bounding boxes recomputed and R1 inputs rendered again for exact equality. This is same-renderer replay, not a second proposal algorithm. Four private contact sheets were actually viewed, covering fixed center slot2 for all32 sources (32 of128 windows). They contain whole objects, parts, background and broad regions; this is display review, not semantic correctness certification. Private media are not published. 580 summary/contrast fields are independently reaggregated; official/independent metric maximum absolute difference 1.44e-15. The early LR/factor scorer retained stale contrast names and emitted empty contrasts; original rows/means were correct. Original artifacts are preserved, and separate corrected contrast tables are published. No GPU rerun or metric value replacement was needed for this export-dispatch correction.

No CURRENT promotion, new detector/segmenter, GT Oracle fit, old-queue restart, or full Table1 run. Per-domain tuning is disclosed development selection, not label-free hyperparameter discovery. Reference agreement is not an identity guarantee. All negative confirmation results are retained without reopening selection.
