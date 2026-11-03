# Fixed128 three-strategy spatial accuracy and actual CPU processing cost

Only this spatial comparison is active under the latest user scope. All earlier completed experiments are reused, not rerun; fulltarget acquisition, end-to-end timing, density and other optional experiments are paused. Latest user explicitly cancelled manuscript work. Same original128query/128distinctparents per direction, same8expert boxes/masks/physicalpositions, same original Source feature layer/resolution/preprocessing and same Current event interval. Source-only Matching parameters remain temperature.05/topK10/context20. No new accuracy predictions/GT scoring/model/Source/expert/GPU/optimizer or Current method change. Real algorithm replay below is required timing, not an accuracy rerun.

The official-kernel/source-domain/positive-control/cold-array checks and both target pilots are already complete. This is a Source-feature VideoWalk module adapter, not the original learned encoder or a strong-baseline claim: it is weaker than pure interpolation on source and target. Current retains its previously audited E/expert soft-candidate behavior at observed frames; Pure and Matching exactly retain raw expert anchors. The comparison controls inputs/time/features; it must not be described as fixing Current anchor outputs or validating an intended hard-anchor implementation. No clamp was silently added.

Timing starts with CPU Source features/boxes, native-grid expert masks/boxes/positions and the common event interval ready, and ends with an owned CPU trajectory. Pure measures interpolation and endpoint holding. Matching measures its actual normalization, two-direction propagation and native-resolution mask-to-box readout. Current measures actual shared E initialization, SourceEvidence construction, complete candidate/score/link rebuilding and final interval DP; no cached E/geometry/path is substituted. Source/ReferDINO/decoding/feature extraction/temporal posterior, experimental reference checks/SHA/saving/resource guards are outside this spatial window. Original intrinsic implementation checks remain part of each function. These clocks are NOT end-to-end deployment latency or component-clock subtraction.

Three actual repeats per query; deterministic forward/reverse cyclic arm order gives each strategy each position once. Torch/BLAS/OpenCV one CPU thread, cores14-15/nice19/idleIO, input/output arrays bitwise equal to old reference on every call, six warmups excluded. Query-internal means precede queryequal128parent10kbootstrap; p90 is the distribution of128query means. Shared-host conditions and low-priority scheduling limit extrapolation to throughput/other hardware. Source/expert inherited cost is nonzero.

|Dataset|Strategy|vIoU %|R.3 %|R.5 %|Unobserved IoU %|Spatial mean ms|95% CI mean ms|p90 query mean ms|
|---|---|---:|---:|---:|---:|---:|---|---:|
|hc2|PureExpert|29.346216|45.312500|17.187500|47.870415|0.423121|[0.416103,0.430585]|0.468759|
|hc2|Matching|24.995983|36.718750|17.968750|39.570804|344.802981|[331.776425,359.862778]|408.314204|
|hc2|Current|31.791298|50.781250|22.656250|54.637004|165.627236|[164.125388,166.959562]|172.850735|
|vidstg|PureExpert|14.912114|17.968750|8.593750|35.917760|0.432491|[0.425558,0.439794]|0.482424|
|vidstg|Matching|10.383367|12.500000|6.250000|22.853901|551.743546|[485.935059,617.271620]|1147.502602|
|vidstg|Current|15.339259|19.531250|10.156250|38.207575|259.876880|[233.696078,285.427763]|486.365193|

|Dataset|Cost comparison|Mean difference ms [95% CI]|Ratio of means [95% CI]|
|---|---|---|---|
|hc2|Current-Matching|-179.175746 [-194.210475,-166.287811]|0.480353 [0.460202,0.498965]|
|hc2|Current-PureExpert|+165.204114 [+163.705494,+166.538035]|391.441352 [384.135346,398.511062]|
|hc2|Matching-PureExpert|+344.379860 [+331.353137,+359.442389]|814.903084 [782.448831,851.909433]|
|vidstg|Current-Matching|-291.866667 [-332.492034,-250.978119]|0.471010 [0.457894,0.486235]|
|vidstg|Current-PureExpert|+259.444388 [+233.270787,+285.000453]|600.883342 [541.626155,660.351357]|
|vidstg|Matching-PureExpert|+551.311055 [+485.497504,+616.838564]|1275.732978 [1125.748453,1427.054869]|

All per-call costs and three-repeat per-query averages are public scalar TSVs. Existing accuracy confidence intervals/positive-negative/severe/rescues/shortlong/distance reports are reused and linked in the original matching pilot, with no new official scoring. No significance-selected sample/parameter changes, equivalence/unseen-test or universal propagation superiority claim. No manuscript edits, extra strategy, end-to-end benchmark, density or fulltarget Source collection was performed.
