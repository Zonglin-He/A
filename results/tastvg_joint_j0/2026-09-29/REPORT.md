# J0 — Final Integrated Online Method

Final-method success criterion met: **False**. Component choices and implementation are locked; no further mechanism rescue. Fixed Fast temporal critic reranking + Slow persistent spatial Rank-RKL. Four arms, same25%availability positions; no added module or tuning.

## Whole-stream four-arm comparison

Metrics in percent. For corruption, average five conditions within each parent then macro over16sources/80cells. Clean16cells shown separately.

| Arm | Corruption sIoU | Corruption tIoU | Corruption vIoU | Clean sIoU | Clean tIoU | Clean vIoU |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 45.350867 | 35.040795 | 15.845411 | 45.848947 | 36.175360 | 16.743969 |
| Fast-only | 45.350867 | 35.374475 | 15.773407 | 45.848947 | 37.148873 | 16.737815 |
| Slow-only | 45.422650 | 35.040795 | 15.881107 | 45.929756 | 36.175360 | 16.783313 |
| Final | 45.422650 | 35.374475 | 15.809093 | 45.929756 | 37.148873 | 16.777270 |

## Prespecified comparisons

All differences in percentage points with source-bootstrap95% CI,10000resamples seed20260929. Intervals conditional on this short fixed stream and development cohort.

| Comparison / subset | Corruption ΔsIoU | Corruption ΔtIoU | Corruption ΔvIoU |
|---|---:|---:|---:|
| fast_minus_frozen / expert | +0.000000 [+0.000000, +0.000000] | +1.334720 [-1.314152, +5.712840] | -0.288018 [-0.621280, +0.001670] |
| final_minus_fast / nonexpert | +0.078608 [+0.022241, +0.143997] | +0.000000 [+0.000000, +0.000000] | +0.047181 [+0.007487, +0.095654] |
| final_minus_frozen / all | +0.071783 [+0.024986, +0.124794] | +0.333680 [-0.362073, +1.428426] | -0.036318 [-0.164640, +0.056689] |
| final_minus_fast / all | +0.071783 [+0.024986, +0.124794] | +0.000000 [+0.000000, +0.000000] | +0.035686 [+0.005630, +0.073389] |
| final_minus_slow / expert | +0.000000 [+0.000000, +0.000000] | +1.334720 [-1.314152, +5.712840] | -0.288055 [-0.621280, +0.001676] |
| interaction / all | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | -0.000009 [-0.000030, +0.000003] |

Expert subset is20cells/4sources; nonexpert60cells/12sources; whole stream80cells/16sources. The three initial nonexpert arrivals before first nonempty spatial evidence remain included. Small expert source count makes immediate-benefit intervals especially limited.

**Integration identity:** Final and Slow-only share identical spatial state and pre-update central boxes. Temporal reranking is readout-only and only active on scheduled arrivals. Consequently, Final−Fast at nonexpert arrivals is exactly the previously measured S1.1 Rank−Frozen effect. J0 verifies correct composition and whole-stream utility; it is not independent new evidence of transfer, nor a demonstration of positive interaction/synergy.

## Clean control and per-corruption future transfer

| Condition | Final−Fast nonexpert ΔsIoU | Final−Fast nonexpert ΔvIoU | Final−Frozen all ΔvIoU |
|---|---:|---:|---:|
| clean | +0.084098 [+0.026113, +0.152190] | +0.051306 [+0.010724, +0.101671] | +0.033301 [-0.142794, +0.186709] |
| frame_drop_5 | +0.058522 [+0.013869, +0.109657] | +0.032990 [+0.003576, +0.067699] | -0.063766 [-0.201293, +0.038911] |
| frame_freeze_5 | +0.080811 [+0.026051, +0.144357] | +0.047882 [+0.010514, +0.092646] | -0.056014 [-0.232699, +0.057842] |
| motion_blur_5 | +0.085730 [+0.021519, +0.161037] | +0.053877 [+0.008529, +0.110134] | -0.028960 [-0.160114, +0.066573] |
| occlusion_5 | +0.080559 [+0.020068, +0.148225] | +0.045625 [+0.006848, +0.093584] | -0.005246 [-0.076538, +0.056614] |
| exposure_5 | +0.087417 [+0.025094, +0.163066] | +0.055531 [+0.010698, +0.112210] | -0.027605 [-0.158643, +0.068631] |

Clean control shows whether benefits are general refinement; the measured gains do not establish corruption-specific recovery. All tested conditions use the original fixed seed0 burst pixels, not a stronger benchmark.

## Current output and state invariants

At every arrival, run the current spatial policy. At expert arrivals, derive up to8native temporal candidates under those same current parameters, use cached frozen UniversalVTG proposals as scalar evidence, and output one student interval. Preserve current pre-update spatial boxes. Separately regenerate9spatial probes, use Sa2VA rewards only to rank them, and apply one SGD.005 reverse-KL update to1792parameters. Carry the update to future arrivals. At nonexpert arrivals, neither expert is read and no update occurs.

Final execution is exact to the sealed Slow-only control for all192pre/post states and 32256gradient coordinates. All96reported spatial outputs match pre-update central boxes; 18/18updated post boxes differ and are excluded from current arm outputs. 0/24scheduled current temporal supports differ from frozen supports; 0/24selected current intervals differ from Fast-only. This is measured, not an assumption that spatial learning cannot affect temporal candidates.

Independent audit: 32256SGD coordinates, 18KL reconstructions, 384temporal candidate scores, 384dual-metric calls,4learned full-output reinsertions and2learned all-six-layer temporal reinsertions.3CPU contracts and8550public scalar checks passed. All four-arm predictions sealed before reading the same16previously exposed GT keys.

## Specialist budget and resources

| Arm | Temporal logical calls | Spatial logical calls | New expert inference in this cached run |
|---|---:|---:|---:|
| Frozen | 0 | 0 | 0 |
| Fast-only | 24 | 0 | 0 |
| Slow-only | 0 | 24 | 0 |
| Final | 24 | 24 | 0 |

The25%availability schedule is shared. Total calls differ: Final uses both experts on24arrivals, single-branch arms one expert. Final vs Fast has the same temporal budget but additional past spatial evidence. Cache reuse saves execution cost; it does not imply equal deployment latency/cost.

New GPU process time 59.100831s, including 1retained failed attempt. New96Final arrivals,216spatial probes,18backward steps; Frozen/Fast/Slow288control arrivals reuse compatible sealed predictions or frozen candidates. No new encoder capture, expert inference or model download. Initial attempt failed at first CPU-vs-CUDA baseline comparison before any saved arrival or effective update. Validation now compares on CPU; original runner/log/lock retained with append-only IMPLEMENTATION_REVISION. No change to method or evaluator.

## Decision and scope

Freeze the research recipe if the prespecified signs above hold; no giant spatial gain or synergy required. Close Norm, temporal OPD, parameter-space OPD, extra support, gates and tuning. Preserve all historical positive/negative results. Recipe configuration and implementation hashes are frozen separately from the historical CURRENT_METHOD production registration. Larger/fresh same-domain corruption evaluation, clean control, secondary cross-domain and alternate streams are the next evaluation phase, not executed by this bounded J0 experiment.

Data and uncertainty: same16repeatedly exposed VidSTG parent sources, onequery each, original Vid-source TA checkpoint5ab12c86, fixed six short streams; main future12sources, immediate expert4sources. This result supports the selected asymmetric recipe on the measured development setting. It does not establish a universal temporal-selection/spatial-adaptation dichotomy or general necessity of constraining specialists to student support.

## Why the sparse Fast arm differs from the prior positive temporal result

Read-only partition of already scored C3 results on exactly the same16sources and five5%corruptions. This is a historical full-availability reference, not an added J0 arm or a new experiment.

| Historical temporal rerank subset | Sources | ΔtIoU (pp) | ΔvIoU (pp) |
|---|---:|---:|---:|
| all | 16 | +6.130562 | +3.156181 |
| scheduled | 4 | +1.334720 | -0.288018 |
| unscheduled | 12 | +7.729176 | +4.304248 |

The locked quarter of arrivals has negative vIoU reranking gain even though the historical full16source reference is positive. In corruption source means: Q01 gains7.965456pp tIoU but zero vIoU; Q05 loses.828373pp vIoU; Q09 loses.327039pp; Q13 gains.003339pp. The unavailable12sources contain the large historical temporal gains, but J0 correctly does not read their specialists. Candidate vIoU oracle headroom on scheduled sources remains+3.585205pp; this is offline GT diagnosis, not deployable benefit. The result does not uniquely establish a bad random schedule or a universal critic failure. Schedule, expert and candidate family remain unchanged.

**Decision:** the positive future-transfer comparison passes, but whole-stream Final−Frozen does not meet the requested positive-sign criterion. Do not label J0 as a successful final-method freeze. Preserve the specified component recipe and stop mechanism optimization; do not choose a favorable schedule with these labels or silently switch to100%expert coverage.
