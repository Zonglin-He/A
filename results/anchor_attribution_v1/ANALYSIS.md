# Attribution audit: direct observations, shared correction, spatial processing and time feedback

The existing NoLocal/Independent results are conditional final-readout tests. Both still use Full auxiliary geometry/DP/pooled features to form Full time. NoLocal additionally forces raw expert boxes at active observations, while Full may select initialization. These results do not identify removal of an entire component throughout the spatial-to-temporal path.

Two new fixed native-time controls use only sealed official caches: Direct8 overwrites the original eight sampled positions on Source with the original expert boxes; SharedDirect8 does the same on shared initialization. All eight observed outputs are identical across these two controls; unobserved outputs are exactly Source and initialization, respectively. No new model, expert, GPU, optimization or tuning. All12205 predictions were sealed before independent CPU labels. Existing full sets are exposed, not a new holdout.

| Stage | HC2 complete vIoU% | Vid complete vIoU% |
|---|---:|---:|
| Source | 24.994752 | 11.777655 |
| Direct8 | 25.656150 | 12.364254 |
| SharedDirect8 | 26.347802 | 13.737897 |
| OursS | 26.811778 | 14.030472 |
| Full | 27.218099 | 14.383314 |

| Fixed-order contrast | HC2 v delta pp [95% CI] | Vid v delta pp [95% CI] |
|---|---|---|
| Direct8-Source | +0.661398 [+0.524506, +0.800363] | +0.586598 [+0.506740, +0.672564] |
| SharedDirect8-Direct8 | +0.691652 [+0.465093, +0.925577] | +1.373643 [+1.111242, +1.646033] |
| OursS-SharedDirect8 | +0.463976 [+0.184904, +0.759000] | +0.292575 [+0.184881, +0.401468] |
| Full-OursS | +0.406321 [+0.210183, +0.603639] | +0.352842 [+0.245056, +0.466626] |

The largest step in Vid is shared geometric correction after matching direct observations (+1.373643pp), rather than propagation/path processing alone (+.292575pp). HC steps are +.661398 direct observations, +.691652 shared correction, +.463976 spatial processing and +.406321 temporal feedback. These are fixed-order stage differences, not independent causal percentages or a unique interaction-free attribution.

Beyond the identical direct8 baseline, OursS increases unobserved sIoU by +1.103415pp in HC and +.780651pp in Vid. OursS observed sIoU decreases by3.394481/4.010830pp versus direct8, because the path is allowed to choose initialization instead of the raw observed expert box (and gates outside its predicted event). This identifies an observed-frame output-policy tradeoff, not proof that the expert always wins or a reason to choose per query using GT.

A genuinely coupled module design is a spatial×temporal2x2: Base=SharedDirect8 with native time; Base+S=current OursS; Base+T=feedback re-pooled on the Base auxiliary track, final Base spatial outputs at the new interval; Base+S+T=current Full. Shared correction and8 evidence are held in all four. The missing Base+T requires different ROI features; compact full caches contain only Full-pooled vectors, so they cannot be relabeled or reused as Base-pooled features. A separate coupled run was subsequently started using the original appearance prefix, then cancelled by the user after60HC predictions, before any complete seal or scoring. It has no scientific result and will not be resumed. No teacher/model/parameter change occurred. Existing readout ablations remain retained with narrower scope.

All signed metrics and severe declines are retained. This audit changes interpretation and controls, not the production method or its frozen full results.
