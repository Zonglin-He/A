# Algorithm pruning: spatial SimplePath and temporal NoSplit

## Scope and fixed candidates

This is actual algorithm-step removal, not a paper rewrite or an equivalent-code refactor. Keep shared initialization, two-sided complete-box propagation/affine validity, exact path DP, native local visual features, native temporal prior, regularized event/background prototypes and one posterior correction. No new expert, observation, encoder, training or hyperparameter scan. Current full remains unchanged.

Spatial SimplePath removes cycle reliability/screening/weights, extent RBF, initialization deviation penalty, and transported adjacent mask-support links. Unary is mean observed-anchor appearance cosine; pairwise is adjacent complete-box IoU. Both nearest past/future anchors, original bidirectional feature matching/affine fit/validity and E fallback remain. Observed/unobserved .5/.5 balancing and .25 link remain. The direct constructor does not compute then discard the removed terms.

Temporal NoSplit removes opposite-offset fitting: one centering/normalization and one event/background prototype pair from all video nodes. Kappa64/beta1/one round/context1.5/native prior/physical MBR and original spatial geometry remain. This reduces the number of prototype pairs from2to1 but includes each frame in its own soft reference, and changes effective fit support while keeping kappa fixed. It is not an isolated causal test of self-inclusion versus evidence scale.

Spatial uses67previously exposed availability-HC inputs with full current time and Native2D pools held fixed. It does not measure the complete changed spatial-ROI-to-time loop. Temporal uses the already exposed128parents/direction and executes spatial DP at the new interval. The two candidates are not combined, are not fresh confirmation, and do not qualify a new full. All323predictions sealed before independent official-label CPU scoring.

## Spatial results

| Arm | Complete vIoU% | Unobserved sIoU% | Observed sIoU% |
|---|---:|---:|---:|
| Source | 27.535844 | 50.047065 | 52.196678 |
| Current | 31.211470 | 55.087238 | 61.051451 |
| BoxLink | 30.985650 | 54.783276 | 60.752490 |
| SimplePath | 31.227264 | 55.295304 | 59.895856 |

SimplePath-Current: v+.015794ppCI[-.407221,+.452530], unobserved-s+.208066CI[-.564453,+1.001885]. These CIs do not prove improvement or equivalence. Observed-s-1.155595CI[-1.903555,-.442955] is a measured cost, not omitted from the tail audit. SimplePath-BoxLink v+.241614/unobserved-s+.512028, CIs span0.

| Baseline | Severe complete-v | Severe unobserved-s | Union(v or unobserved-s) |
|---|---:|---:|---:|
| SimplePath-Current | 1 | 3 | 3/67 |
| SimplePath-Source | 11 | 11 | 13/67 |
| Current-Source | 11 | 12 | 14/67 |

Severe means >5pp loss. SimplePath has6observed-s severe losses versusCurrent. Including observed-s, complete-v and unobserved-s together changes the union; see observed_tail_audit.json. Metric counts overlap and must not be added.

| Group | N | SimplePath-Current v pp | Unobserved s pp | Observed s pp |
|---|---:|---:|---:|---:|
| short_le5s | 19 | -0.599009 | -0.327467 | -1.811653 |
| long_gt5s | 48 | +0.259154 | +0.420048 | -0.909573 |

## Temporal results

| Direction/panel | Current t% | NoSplit t% | Delta t pp | Current v% | NoSplit v% | Delta v pp |
|---|---:|---:|---:|---:|---:|---:|
| hc2/all128 | 49.189609 | 49.830647 | +0.641039 | 28.403571 | 28.705711 | +0.302139 |
| hc2/exposed_dev96 | 50.422217 | 50.842869 | +0.420652 | 29.326517 | 29.546576 | +0.220058 |
| hc2/exposed_confirm32 | 45.491785 | 46.793983 | +1.302198 | 25.634734 | 26.183116 | +0.548382 |
| vidstg/all128 | 40.798627 | 40.745963 | -0.052664 | 19.833162 | 19.789413 | -0.043750 |
| vidstg/exposed_dev96 | 41.432939 | 41.606102 | +0.173163 | 20.730961 | 20.784660 | +0.053699 |
| vidstg/exposed_confirm32 | 38.895691 | 38.165546 | -0.730145 | 17.139766 | 16.803670 | -0.336096 |

HC128 NoSplit-Current t+.641039CI[+.217850,+1.067246], v+.302139CI[+.029837,+.574166]; Vid128 t-.052664CI[-.430708,+.262396], v-.043750CI[-.255662,+.130100]. Parent-cluster intervals use query-equal paired differences.

| Direction | Current-Source t pp | NoSplit-Source t pp | Current-Source v pp | NoSplit-Source v pp | Source-severe time Current->NoSplit | Source-severe t/v union Current->NoSplit |
|---|---:|---:|---:|---:|---|---|
| hc2 | +1.274417 | +1.915455 | +3.393524 | +3.695664 | 6->11 | 19->23 |
| vidstg | +1.809039 | +1.756375 | +4.188210 | +4.144461 | 4->9 | 21->25 |

NoSplit is an actual simpler temporal algorithm with a mean tradeoff, not a claim of improved damage control. HC Source-severe union19->23 and Vid21->25; Vid exposed-confirm32 t-.730145/v-.336096 versusCurrent, severeSource union6->9. These negatives prevent treating a pooled mean as a universal gain.

| Direction/group | Current-Source t pp | NoSplit-Source t pp |
|---|---:|---:|
| hc2/short_le5s | +0.460529 | +1.225705 |
| hc2/long_gt5s | +1.502305 | +2.108585 |
| vidstg/short_le5s | +0.233212 | -0.006256 |
| vidstg/long_gt5s | +3.243746 | +3.361158 |

## Real CPU cost

| Spatial core | Mean per-query median ms |
|---|---:|
| Current | 86.740355 |
| BoxLink | 56.744817 |
| SimplePath | 57.205404 |

Original86.740ms->SimplePath57.205ms,34.05%less core CPU time. BoxLink56.745ms is similar toSimplePath; deleting extent/guard reduces the mathematical/scoring machinery but does not demonstrate extra speed beyondBoxLink. Time feedback alone0.984823ms->0.621977ms. Neither supports an end-to-end multiple-times speedup. Model/expert/transfer costs are not measured by this cached component timing.

## Decision

Spatial SimplePath is worth retaining as a concrete simpler algorithm: complete-v and unobserved means are close, with fewer scoring mechanisms and lower core CPU cost; observed-frame loss and all severe cases remain explicit. Temporal NoSplit is also structurally simpler, with HC mean gains and a small pooled Vid mean loss, but has a worse Source-relative severe tail. Keep it as a disclosed engineering candidate, not an automatic production replacement. Do not tune its shrinkage or add gates to hide the failures. No new full or combined-method result is claimed.

Equivalent cumulative hull/vector-prototype/singleton-initializer/grid-adapter checks are separate implementation simplifications, not substitutes for this algorithm deletion study. Source/caption Method still mentioning six-layer averaging describes the old Global6, which is absent from current production. This is a version correction rather than an additional proposed algorithm deletion.

## Integrity

1548 official-versus-dense metric checks max1.33e-15. All existing Source/Current/Global6/NativeSpatial/BoxLink saved metric dictionaries match exactly; all spatial times fixed. Prediction CPU97.72s, scoringCPU18.62s;0newmodels/RGB/expert/GPU/training. InitialV1 had a leftover removed extent-call NameError before its first prediction; original plan/source/log preserved, V2 has independent fingerprint29847357fe67eca0fc8524e85e83ad1cf7b15bcf0743de00e5e72b7716030381. No hash waiver or prediction migration.

[Protocol](protocol.json), [all rows/groups/CIs](report.json), [all signed comparisons](all_signed.tsv), [all severe comparisons](all_severe.tsv), [observed tail audit](observed_tail_audit.json). Private media/captions/GTcoordinates/weights/features/masks remain private.
