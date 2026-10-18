# Sparse anchors and a constant four-parameter error state

Correct sparse references support a useful low-dimensional geometric correction on this exposed32-query panel. The corresponding actual DINO references do not satisfy the two-domain unobserved-decoder criterion. This is a cached output intervention, not a new Student, online TTA run, identity model or deployment result. No OPD, filtering, smoothing or confirmation is launched.

## Question and matched controls

The fixed time component is final-distribution MBR, selected by the user as an engineering choice. Its earlier T0 replacement-gate failure, HC fast-motion risk and old MAP/projection comparisons remain valid. The present spatial experiment does not retrospectively promote MBR. Native, projection and merged MAP remain sensitivity clocks for the same sealed spatial outputs; no correction is re-estimated per clock.

Each baseline is the original saved query-start Before, with its actual inherited LN effect; frozen Source is a separate sensitivity. Base keeps every original box. LocalAnchor replaces only real reference decoder nodes in normalized cxcywh, then uses the original benchmark xyxy interpolation. GlobalError fits one equal-weight residual mean in z=(cx,cy,logw,logh), and adds that constant to every original decoder node. It has four parameters per query: center translation and independent width/height scaling. There is no clipping, robust estimator or speed-dependent rule; there is no neural-network training or Student parameter update. No-reference interventions are exact Base no-ops.

LocalAnchor is not the old source24 event-relative32-knot minimum-L2 equality projection. That method constrained probability-space knots touched by W; this one edits actual decoder nodes. The old identity-trajectory proposal interpolated varying logit(cxcywh) residuals and fed stochastic OPD, also a different operation. These controls are not numerical replays of those old mechanisms.

## Reference conditions and primary effects

DINO_full uses all original admitted Top1 boxes and was sealed for all32 before this experiment opened oracle GT. GT_planned uses exact annotations at the original four planned nodes. GT_common and DINO_common use the identical admitted-node subset with exact GT availability. DINO_common therefore uses label availability and is diagnostic only. The oracle loader parses full truth files, but prediction generation accepts only sealed sparse targets; held-out dense GT does not enter the correction function. All oracle/common predictions were sealed before dense scoring.

| Target | References | Count / no-ref queries | Global−Base full Δv pp [95% CI] | Unobserved actual decoder Δs pp [95% CI] | Global−Local full Δv pp [95% CI] |
|---|---|---:|---:|---:|
| hc2 | GT_planned | 49 / 1 | +8.962132 [+1.182601, +17.904069] | +9.561532 [+0.738305, +19.172183] | +3.339321 [-3.999100, +11.681834] |
| hc2 | GT_common | 43 / 1 | +9.189653 [+1.440874, +18.132330] | +10.006148 [+0.808964, +19.953722] | +4.648867 [-3.079517, +13.141186] |
| hc2 | DINO_common | 43 / 1 | +1.434991 [-6.775416, +10.989518] | -1.949017 [-15.568376, +11.085680] | +1.097365 [-5.806937, +9.280183] |
| hc2 | DINO_full | 56 / 0 | +0.689673 [-7.435740, +10.447267] | -2.675767 [-16.059301, +10.133904] | +0.348842 [-6.702702, +8.759932] |
| vidstg | GT_planned | 30 / 4 | +9.344504 [+4.147196, +15.341058] | +23.409595 [+9.768638, +38.201398] | +7.472907 [+3.149781, +12.375366] |
| vidstg | GT_common | 24 / 6 | +9.248077 [+3.918560, +15.287773] | +23.100538 [+9.432097, +37.916999] | +7.664849 [+3.213439, +12.648112] |
| vidstg | DINO_common | 24 / 6 | +1.969260 [+0.573286, +3.754948] | +6.427391 [-0.352998, +16.734992] | +1.581410 [+0.391302, +3.192663] |
| vidstg | DINO_full | 50 / 2 | +1.931277 [+0.646712, +3.470272] | +6.176844 [-0.629624, +15.290373] | +1.542098 [+0.466236, +2.880126] |

GT_common passes the descriptive positive-mean ability screen for full vIoU and unobserved actual decoder sIoU in both targets. Its full Global−Local gain is +4.648867pp in HC with a CI crossing zero, and +7.664849pp in Vid with a positive CI. Correct-reference diagnostic gains are not deployable effects. Even GT_common has4 negative full-v and5 negative unobserved-s parents in HC, and1 negative unobserved parent in Vid; the constant model is not universally correct.

DINO_common fails the HC unobserved mean (−1.949017pp); DINO_full also fails it (−2.675767pp). Both HC CIs are wide and cross zero, so these are not proofs of statistically established harm. They fail the prelocked mean screen. Full DINO improves mean full vIoU by+.689673pp HC and+1.931277pp Vid, but that is insufficient when the required unobserved endpoint fails. The matched-reference gap supports reference reliability as a limitation; it does not distinguish identity mistakes from extent/localization errors, and does not rule out limitations of the constant correction model.

All16 queries per domain remain. Reference counts HC/Vid are planned49/30, common43/24, full DINO56/50. GT_common/DINO_common have1/6 no-reference queries; full DINO has0/2. Zero-reference cases and unknown supports are retained, never dropped or assigned fabricated GT.

## Actual decoder support, interpolation and tails

The primary known unobserved decoder support excludes all four original planned positions for every arm/reference condition, including unadmitted or annotation-missing slots. Within-MBR, observed, all-known and far support (at least one quarter of original input duration from every planned anchor) remain separate. Benchmark interpolation uses annotated non-decoder frames and retains original no-extrapolation behavior. Far support is available for only4 HC and7 Vid parents; empty supports remain null with counts.

LocalAnchor has exactly the same IoU as Base at unobserved actual decoder nodes by construction. Its benefit at benchmark-interpolated frames can still be real. This design invariant does not show that all interpolation methods are ineffective. Full Δv is independently decomposed into observed nodes, unobserved decoder nodes and benchmark-interpolated frames, with exact sums for all512 arm/reference/base cases.

HC DINO_full has a large positive H016 full-v change of+57.114365pp. Removing that saved score from the mean gives−3.071973pp; the primary mean still includes it. The complete leave-one-parent saved-score range is−3.071973 to+2.480409pp. This is a sensitivity summary, not online replay, refitting or case removal. Vid full-v leave-one means stay+1.464592 to+2.116971pp. All signs and severe losses below−5pp are retained.

HC full DINO fast-motion Q4 has6 parents, mean full Δv−5.293275pp and unobserved Δs−5.229923pp; Vid fast has3 parents. HC has no common Q1 slow group in this panel. These small/absent strata do not justify a general fast/slow routing claim. Target size, event length and query-form bins, counts, risks and CIs are all retained.

## Does a constant correction preserve useful motion?

The correction preserves pre-metric center increments and log-size increments exactly. That is algebra: it does not show correct identity, event timing or globally smooth prediction error. In the Before baseline, correction-error center speed is below GT center speed for only5/16 HC and10/16 Vid parents. Median error/GT speed ratios are1.176150/.922498. These use normalized-image-center distance per true second, not target-size-relative speed. GT-common benefits can coexist with locally varying or badly estimated errors.

Source sensitivity preserves the main contrast: GT_common unobserved gains+10.685929/+25.307528pp HC/Vid; DINO_common−1.340500/+8.617848, full DINO−2.124377/+9.845364. Before and Source are different fixed starts. Neither is a replay of a new correction carried through the original stream, and the positive Source Vid CIs do not replace the primary Before endpoint.

## Verification, cost and scope

Independent no-GT checking uses only original Source logp0, never actionness values: 170303 final legal candidates, including zero-mass decisions, pass direct physical expected-tIoU comparison (max1.27676e-15); independent two-dimensional CDF reconstruction differs by at most2.72475e-16. All256 reference/base corrections pass separate constant least squares and local-node/no-op/increment checks;42 are exact no-op pairs. There are64 bit-exact original Before/Source checks and53,616 independent aggregate/CI/support/stratum/count checks. The original scorer additionally checks9,728 metric scalars and2,048 time invariants.

DINO_full cached generation took2.155654s including I/O and q/MBR. Oracle generation took0.059577s after the GT loader (loader time excluded), and dense scoring34.795691s. Separate independent DINO numerical checking took31.626337s. These are CPU diagnostic stages, not fresh online end-to-end cost; do not sum them as deployed latency. New GPU/model/expert/training/Student-update counts are0.

The cohort is the original exposed rollback32,16 independent parents per target, with different original cross-domain Source checkpoints and inherited Before trajectories. One query per parent makes query/parent macro means coincide, but inherited-state coupling remains. Fixed-seed10,000-draw parent intervals describe these saved outputs; they do not replay a stream after deleting parents and are not multiplicity-adjusted final confirmation.

The permitted conclusion is geometric ability with correct sparse references and unresolved actual-reference reliability for this constant four-parameter correction. The locked matched/full-DINO two-domain condition fails, so no structured correction, OPD, filter, new expert, parameter training or automatic confirmation follows. CURRENT/MAIN/PAPER and all previous positive and negative evidence remain unchanged.

- [Complete anonymous results, signed parent effects, CIs and strata](RESULTS.json)
- [Locked descriptive decision](DECISION.json) and [actual protocol/cost](ACTUAL_PROTOCOL.json)
- [Independent aggregate audit](ROOT_AGGREGATE_AUDIT.json), [numerical verification](VERIFICATION.json), [saved-score and speed diagnostics](DIAGNOSTICS.json)
- [Aggregate figure](ERROR_STATE.svg) and [protocol](../../../../protocols/stvg_error_state32_v1.md)

No GT coordinates, original query/parent text, media, q/utilities, weights, raw tensors or personal records are exported. Oracle/common conditions are explicitly diagnostic; DINO_full only avoids GT in generation, and has not passed the measured deployment-benefit screen.
