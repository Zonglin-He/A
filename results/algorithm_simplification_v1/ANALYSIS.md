# Algorithm simplification: exact temporal formulas and a bounded spatial alternative

## Decision

The temporal kernel can be written as two prototype vectors, and the two-offset hull can be computed by positive cumulative sums. These are algebraic rewrites, verified against the existing method; the frozen production files remain unchanged. The structural spatial candidate reduces CPU work with a modest average accuracy loss and additional severe per-query losses. It is an engineering tradeoff observed on this exposed HC availability panel, not an equivalence result or a new production method.

The independently checked [reference.py](../../studies/algorithm_simplification/reference.py) shows the temporal core without model/runtime/archive machinery. User clarified the scope is code simplification with core components unchanged; the structural alternatives below are retained as completed diagnostics, not proposed replacements. Further simplification work is restricted to equivalent calculations.

## Locked comparison and evidence

One predeclared structural candidate, NearestBoxLink, retains the original forward/reverse feature correspondences and axis-wise complete-box affine transform but uses only the physically nearest observation per unobserved frame (ties earlier). It removes cycle screening/weighting and replaces transported adjacent mask support with complete-box IoU. Invalid nearest fits fall to initialization, with no farther-anchor retry. The BoxLink control keeps both nearest past/future observations; NoCycle is the exact previously sealed ablation. Original candidate support, guard, score coefficients, legality checks and exact DP are retained. No weights, thresholds, anchor counts or feature budgets are scanned. Controls are not promoted after seeing results.

The same67queries/67parents have exact-official HC metadata and complete historical raw Source/mask caches. Membership was fixed by cache availability before labels, not outcome. It is neither fullHC1902 nor a fresh confirmation. All67 predictions sealed before a separate CPU process read fixed official labels. Every spatial arm fixes the current Full temporal interval, Native2D pools/prior and eight expert observations. Therefore this estimates a spatial replacement, not an end-to-end altered spatial-ROI-to-time feedback system. No Vid spatial result is claimed.

## Spatial precision

| Arm | Fixed unobserved sIoU% | Complete vIoU% | tIoU% | R@.3% | R@.5% |
|---|---:|---:|---:|---:|---:|
| Source | 50.047065 | 27.535844 | 54.180620 | 47.761194 | 14.925373 |
| Current | 55.087238 | 31.211470 | 54.460918 | 49.253731 | 17.910448 |
| NoCycle | 54.868079 | 31.060033 | 54.460918 | 53.731343 | 17.910448 |
| BoxLink | 54.783276 | 30.985650 | 54.460918 | 55.223881 | 16.417910 |
| NearestBoxLink | 54.487413 | 30.810557 | 54.460918 | 52.238806 | 17.910448 |

| Comparison | Delta unobserved s pp [95% parent CI] | Delta complete v pp [95% parent CI] | Severe v / s / union |
|---|---|---|---|
| NoCycle-Current | -0.219159 [-0.901584, +0.408188] | -0.151437 [-0.767830, +0.361950] | 1 / 4 / 4 |
| BoxLink-Current | -0.303962 [-1.313378, +0.580303] | -0.225820 [-0.879485, +0.318269] | 2 / 5 / 5 |
| BoxLink-NoCycle | -0.084803 [-0.784610, +0.542827] | -0.074383 [-0.455825, +0.287751] | 1 / 4 / 5 |
| NearestBoxLink-Current | -0.599825 [-1.486774, +0.238504] | -0.400913 [-0.972369, +0.140404] | 4 / 7 / 9 |
| NearestBoxLink-BoxLink | -0.295863 [-1.305674, +0.717997] | -0.175094 [-0.931153, +0.581918] | 3 / 6 / 7 |
| NearestBoxLink-Source | +4.440348 [+0.727433, +8.297220] | +3.274713 [+0.841048, +5.732237] | 10 / 12 / 14 |
| Current-Source | +5.040173 [+1.031276, +9.281340] | +3.675626 [+1.143120, +6.271369] | 11 / 12 / 14 |

Severe is a decline **greater than5pp in complete vIoU OR fixed unobserved sIoU**, counting each query once; this is not the temporal t/v union. NearestBoxLink creates9/67 severe losses versusCurrent, although its Source-relative severe union is14/67, equal toCurrent14/67. Mean CIs spanning zero do not prove equivalence. NearestBoxLink complete-v has30wins/30harms/7ties versusCurrent; unobserved-s has29wins/31harms/7ties. The Source-relative improvements are v+3.2747pp and unobserved-s+4.4403pp.

| Group | N | Nearest-Current unobserved s pp | Nearest-Current v pp | Severe union vsCurrent |
|---|---:|---:|---:|---:|
| short_le5s | 19 | -0.710784 | -0.119923 | 2 |
| long_gt5s | 48 | -0.555904 | -0.512139 | 7 |

Full group tables for every arm, both metrics, every signed query and all negative tails are retained in report.json/signed.tsv/severe.tsv.

## Actual CPU component cost

Same cached tensors, one warmup and three alternating-order repeats per query; means below are across each query's median. T=65, actual gridN63–70. No original geometry is built and discarded in either lighter constructor.

| Component | Mean ms/query |
|---|---:|
| Current_core | 89.121997 |
| BoxLink_core | 58.696170 |
| Nearest_core | 34.407799 |
| Original_grid_adapter | 42.913734 |
| Known_grid_adapter | 6.275742 |

Nearest spatial core takes 38.61% of original time (61.39% reduction). Grid-adapter plus core is 132.036->40.684ms, a 69.19% component reduction. Explicit source H/W removes positional-grid factor search/validation; extracted features and valid masks are bitwise identical on all67. Deployment adapters must obtain H/W from the actual native encoder, never GT. This is a cached CPU component result, **not an end-to-end speedup**.

Original profiles put anchor affinity at31.19ms and affine fits at14.59ms, nested in geometry72.28ms. Cycle/link Dice is1.89ms, also nested. Deleting Dice alone cannot explain large total savings; the main structural reduction is source-target matching: original208–228anchor affinities+64adjacent affinities, Nearest114anchor affinities+0adjacent. The nearest source may be before or after the target; it still uses both directional correspondence matrices. Adjacent DP itself is about1–2ms and has positive HC full ablation contribution; it is retained.

The separate342-query current Vid runtime snapshot has mean Source2242.757s, capture4.026s, evidence+geometry0.217s, Source-toCPU0.427s and Source-toGPU0.263s. Natural visual/RGB reuse and cached expert evidence are present. These overlapping/nested costs are not additive and do not measure uncached resident deployment. They show that CPU spatial simplification alone cannot justify claiming the entire algorithm is multiple times faster. Model residency/transfer changes require a later fair GPU benchmark, not interference with the active full.

## Temporal simplification and exactness

### Two prototype vectors

For centered/L2-normalized fitting vectors a_i and held-out vector b_t, K=(1+b_t dot a_i)/2. Weighted kernel means with shared-mean shrinkage equal (1+b_t dot v+)/2 and (1+b_t dot v-)/2, where v+ and v- are weighted means of a_i shrunk toward their common mean. No T-by-T kernel is needed. Same other-offset fitting, fixed kappa64, zero-contrast behavior, inclusive interval sum, native prior, one update and MBR remain. Real-arithmetic equivalence does not imply every floating-point bit is identical.

### Positive quadratic-time hull

Each physical hull endpoint belongs to exactly one of the two disjoint offset grids. Four nonnegative terms cover the cases: both endpoints fromoffset0; both fromoffset1; start0/end1; start1/end0. Strict-contained mass and row/column cumulative sums compute these cases in O(T²) instead of enumerating O(T⁴) interval pairs. There is no interval pruning, gain, CDF subtraction or negative-mass clipping.

Actual2030-cache audit: full HC1902 plus existingVid128, T16–200. Every native hull matches within1.3184e-16, every updated posterior within1.8042e-16 and every frame feedback within8.8818e-16. **2030/2030 final intervals exactly match their original saved outputs.** With the same immutable spatial geometry and deterministic DP, the final spatial outputs are unchanged aliases; no new label score or spatial-inference claim is made.

| Actual T | Original hull ms | Cumulative hull ms | Original prototype ms | Vector prototype ms |
|---:|---:|---:|---:|---:|
| 16 | 0.020769 | 0.076008 | 0.665576 | 0.588689 |
| 65 | 1.132977 | 0.252715 | 1.076493 | 0.765550 |
| 133 | 33.470482 | 0.801174 | 2.220560 | 1.273623 |
| 200 | 189.986132 | 1.650617 | 2.718229 | 1.750055 |

AtT200 the original method materializes24,502,500 interval-pair weights and their key array, roughly392MB for those two FP64/int64 arrays alone; the new method uses T-by-T tables, not those products. This is especially useful on long Vid grids. AtT16 the cumulative implementation is slower in this microbenchmark; the bounded protocol does not select implementations by outcome.

The independently readable reference implementation additionally matches all67 actual cases, randomT16/33/65/133/200 and exact constant no-op; see REFERENCE_CHECK.json.

## Concrete failure and rescue audit

Post-seal audit replays the already sealed outputs, not a new method arm. Nearest changes752 full-timeline boxes;344 changed nodes lose the old selected candidate because it came from the discarded other anchor, and98 changed nodes have no valid nearest propagation. Counts overlap and are not GT-correctness labels.

| Case | Delta v pp | Delta unobserved s pp | Changed boxes | Old selected candidate absent | Changed nodes with invalid nearest fit |
|---|---:|---:|---:|---:|---:|
| H059 | -6.757579 | -7.492102 | 8 | 4 | 1 |
| H009 | -5.843423 | -6.419300 | 8 | 5 | 4 |
| H027 | -5.134424 | -4.657386 | 7 | 7 | 6 |
| H049 | -3.654543 | -10.002711 | 6 | 2 | 0 |
| H044 | +5.601405 | +6.702307 | 12 | 0 | 0 |
| H061 | +4.260819 | +7.169232 | 12 | 3 | 1 |

H027 loses the old selected candidate at all7 changed nodes and has unavailable nearest fits at6, consistent with a cost of deleting the other anchor. H044 improves5.6014pp complete-v without losing any old selected candidate, showing that altered scoring/links can help even without candidate removal. Multiple components change jointly; these are associations, not proof that one component causes each loss or rescue. Do not tune gates using these cases.

## What to simplify now

1. Use the vector prototypes as the readable mathematical definition. Keep the existing shrinkage, other-offset construction and single update.
2. Preserve the native two-offset prior but compute its hull with cumulative tables in the tested adapter; no new temporal module is needed.
3. Keep exact DP: it is inexpensive and its existing full ablation supports it. Keep complete-box propagation as the spatial mechanism.
4. The nearest-anchor/box-link version is a real, measured smaller algorithm, not an exact replacement. It remains a candidate with all losses disclosed; the better-looking two-anchor control is not silently promoted after the study.
5. Production integration must preserve the currently frozen run. This study changes neither its source/configuration/fingerprint nor its saved results.

## Reproduction and evidence

Independent plan fingerprint3d4f32eb7828d10b0b2b548b8523d4bb3391e3d3b3172c7caddb6a77e922d6cf. Public source: studies/algorithm_simplification. Prepare tests3small hull references, kernel equality and constant no-op; exact math60.94sCPU, spatial replay/timing66.56sCPU, separate score1.46sCPU, independent readable/candidate audit12.41sCPU. No newRGB/model/encoder/text/expert/observation/GPU/optimizer/training. These are research costs, not deployment latency.335 official-vs-dense metric checks max8.88e-16, Source/Current/NoCycle saved metric dictionaries exact, full time equal across spatial arms. Source-only profiling and natural runtime snapshot are separately preserved.

[Protocol](protocol.json), [all metrics/CIs/groups](report.json), [all signed outcomes](signed.tsv), [all severe losses](severe.tsv), [2030 exact math](EXACT_MATH.json), [readable reference and case audit](REFERENCE_CHECK.json), [component profiling](profile_current.json), [natural runtime snapshot](natural_runtime_cost.json), [prepare tests](tests.json). Private videos/captions/GT coordinates/weights/masks/features remain outside Git.
