# A0.3 Shared Channel Basis Feasibility

Status: prospective registration; requested in attachment 5a94937e. Results must be reported separately.

Question: A0.2 established per-query channel low rank, but not a common channel subspace. Test whether one Train-only basis preserves exposed Dev directions, before implementing any factorized predictor.

Use the same sealed A0 Train128 (95 original Vid training parents) and Dev64 (16 already exposed diagnosis parents). Fields derive from frozen official PTD4B/B1 native source-GT oracle gradients and the frozen union256 basis. Source privilege and prior development exposure remain explicit. Do not load PTD, read new labels/pixels, or access fresh31/388 in the CPU stage. No optimizer or network fitting.

For each complete field A[THW,256], use FP64 uncentered C=A^T A/||A||². Every query gets weight 1/128 regardless of length, norm or parent size. Only Train128 enters mean C_train and its descending eigendecomposition. No mean-centering, whitening, or per-parent weighting. Save the full spectrum, covariance and basis before opening Dev fields in this stage. Evaluate fixed k={1,4,8,16,32,64}, a diagnostic curve rather than rank tuning. A single NumPy default_rng seed20260929 Gaussian QR basis (nested columns) is a sanity control, not a candidate. Fix QR signs; scientific subspace measurements are sign-invariant.

Per query retain shared/random energy, flattened cosine=sqrt(energy), the existing complete per-query SVD energy upper bound including rank64, and shared/optimal and random/optimal ratios. Report equal-query split mean/median/min/max, preserving every query. Independently rebuild the Train covariance/basis and full192 projection measurements in Torch FP64; compare subspace projectors at each rank, eigengaps, and raw reduced-projection energy, and verify projection cosine identities. Locked tolerances: scalar/identity absolute1e-10, normalized covariance1e-12, projector1e-8 (if degeneracy causes failure, preserve it and diagnose, never silently redefine rank).

CPU engineering allocation: 900 seconds each for primary and independent readback, four CPU threads, <=256MiB new output, free disk >=8GiB throughout. GPU seconds0; CPU seconds recorded separately. Cumulative GPU allocation/wrapper cap remains null. This stage must pass synthetic unequal-length/scale invariance, projection/norm, split separation, zero-support and threshold controls before actual data measurements.

## Conditional native screen (locked before CPU results)

Trigger iff independently verified Dev64 shared-rank32 median energy >=.75. Failure stops fixed shared-basis route; next candidate is query-conditioned basis analysis only, with no learned model authorized here.

If triggered, serially run only Shared-R16 and Shared-R32 on the same Dev64, frozen PTD/B1/union and original native decoder. Reuse B1 and Full Oracle sealed predictions after hash/input/checkpoint/support verification, no retraining. Project A_k=A B_k B_k^T and normalize each complete projected field back to the original Full Oracle coefficient norm, then map through frozen Q_U at the same insertion/radius. Same correction enters both native passes. No changes in support, native objective, radius, precision recipe, sample list or result-based state selection. An exactly zero projection is an explicit failure, no fabricated direction or dropped query.

Before actual GPU launch, lock runner/inputs/sealed basis and storage estimates; 3600-second engineering allocations at complete-query boundaries, cumulative capnull and >=8GiB disk. One Luna/max read-only watcher per active stage, no 60-second loop. Record all loading/failure/replay/wrapper time, retain partial evidence. Seal all new predictions before offline source scoring. No GT use beyond explicit existing source oracle/diagnostic scope.

Report parent-macro and query t/s/v, paired parent CIs, >5pp query harm, B1-good retention, and Δv_rank/Δv_full as descriptive oracle-gain retention (undefined if denominator0; signed if negative). Native pass means Δv>0, Δt>=0, Δs>=0, and no predefined collapse: Δt or Δs<=-1pp with >=12/16 parents negative. R16 pass has priority; else R32 pass selects rank32 candidate; neither pass stops fixed low-rank correction. Gain-retention ratio alone is not a gate. No extra ranks in native, no learned predictor in this task.

Do not start full618/full447, fresh388, TVG/SVG, trust gate, OPD or target. Neither retained energy nor native oracle gain proves ordinary-input learnability. Preserve CURRENT, historical failures and optimizer integer-key corrections.
