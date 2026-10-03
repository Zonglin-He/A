# Fixed-budget temporal candidate coverage and grid oracle

One replacement allocation completed on both datasets and both existing panels. Eight intervals per expert arrival, the original temporal scorer, fixed A boxes and persistent Uniform updates. This separates an unlabelled readout contrast from privileged candidate/grid oracle diagnostics. Values are dense vIoU percent or percentage points as labelled, with 95% paired source-bootstrap intervals (10,000 draws, seed20261003).

## Matched setting

Each dataset retains the same32 development and16 within-batch disjoint confirmation sources, one query/source, two orders, clean plus five5% transient corruptions and25% scheduled expert arrivals. All sources have historical exposure, including preceding mechanism diagnostics. Confirmation is descriptive, with no result-based generator selection. Total1,152 arrivals;288 expert candidate comparisons. Same-domain official TA-STVG checkpoints and original Paper48 observed pixels/grid. A means the existing Uniform persistent spatial learner plus the original single-view temporal Fast, not Frozen.

Vid K1/lr .033761698432507946/teacher .34902548789596055; HC K8/lr .006097133675874025/teacher1; rho .05/student1/D4/1792. All actual A pre/post states and full spatial boxes are unchanged. No current correction, spatial loss, temporal view or new specialist observation is added.

## The one predefined replacement pool

Slot0 retains current native. Six slots cross early/middle/late center with short/medium duration; the final slot is long at any center. Fixed thirds divide positions and lengths on the physical observed window. Within each slot, frozen source-head endpoint products select the best unused legal interval; exact ties use start then end index. Start/end probabilities are separately normalized within each original offset, then interleaved with equal offset mass. This is an allocation prior, not a calibrated likelihood of the native two-offset envelope. It is independent of UniversalVTG scoring. The preserved native is checked against the source cache rather than substituted.

Both old and new pools use the literal original max(confidence*interval-IoU) critic, its same cached first-view proposals/confidences, and native-first np.argmax ties. New intervals are student-grid readouts, not specialist proposal coordinates. The previous two-view min rule is not used. All new pools/selected readouts across both datasets and both panels were globally sealed before new GT scoring. No new expert, backbone, suffix or backward calls occurred.

## Actual readout: complete corruption streams

| Panel / dataset | Cells / sources | A old8 (%) | New8 actual (%) | New−A vIoU (pp) | New−A tIoU (pp) |
|---|---:|---:|---:|---:|---:|
| search/vidstg | 320/32 | +17.4529 [+11.1003, +24.5945] | +16.4273 [+10.4234, +23.1537] | -1.0256 [-3.2766, +0.3380] | -1.7201 [-4.8448, +0.4641] |
| search/hc2 | 320/32 | +30.9637 [+24.1821, +37.9082] | +29.0016 [+22.8086, +35.2205] | -1.9621 [-4.5365, -0.0595] | -2.7747 [-6.2788, -0.0637] |
| confirm/vidstg | 160/16 | +32.6465 [+20.5681, +45.3322] | +32.0874 [+20.2247, +44.5582] | -0.5591 [-2.5122, +0.6392] | -1.0619 [-4.1053, +1.0323] |
| confirm/hc2 | 160/16 | +26.7630 [+17.9558, +35.3714] | +25.3281 [+16.6512, +33.9805] | -1.4350 [-2.9820, -0.2682] | -3.7274 [-7.5582, -0.8442] |

Nonexpert output is exactly A at all864 positions; full-flow increments are computed from complete source/condition/order rows, not from an assumed25% multiplier. No persistent state is changed, so this measures current temporal readout only, not future parameter transfer.

## Oracle opportunity versus realized selection: corruption experts

| Panel / dataset | Cells / sources | Old8 oracle (%) | New8 oracle (%) | New−old oracle (pp) | Actual new−old (pp) | New selection gap (pp) |
|---|---:|---:|---:|---:|---:|---:|
| search/vidstg | 80/16 | +19.9520 [+8.4787, +33.2333] | +19.4813 [+9.4343, +30.4153] | -0.4707 [-5.4872, +3.9615] | -4.1025 [-12.9466, +1.4573] | +5.7515 [+1.7884, +10.7980] |
| search/hc2 | 80/14 | +35.7327 [+24.6867, +47.4418] | +38.7655 [+29.5005, +48.4167] | +3.0328 [-1.0070, +8.1399] | -8.2321 [-19.0289, +0.0756] | +16.8003 [+8.7658, +26.3038] |
| confirm/vidstg | 40/8 | +25.2270 [+12.4031, +38.6943] | +30.5185 [+16.7186, +44.2284] | +5.2915 [-0.7694, +13.6521] | -2.2366 [-9.7672, +2.4672] | +10.7071 [+2.9602, +19.9074] |
| confirm/hc2 | 40/7 | +37.3005 [+26.6416, +48.2405] | +39.4658 [+26.7620, +53.3025] | +2.1654 [-4.6253, +10.6534] | -6.7675 [-12.0559, -2.4923] | +12.4759 [+5.0137, +20.5997] |

The oracle selects with GT only after sealing and is not an implementable result. Both pools retain native but the new pool does not contain the entire old pool, so its per-cell oracle can decrease; all negative changes are preserved.

## Where the old coverage gap resides

| Panel / dataset | GT-time−old8 oracle (pp) | Grid−old8 oracle (pp) | GT-time−grid oracle (pp) | Grid−new8 oracle (pp) |
|---|---:|---:|---:|---:|
| search/vidstg | +11.0206 [+5.0167, +18.1370] | +9.9246 [+4.2491, +16.4867] | +1.0960 [+0.2559, +2.1349] | +10.3953 [+5.4675, +16.0960] |
| search/hc2 | +17.8843 [+12.0898, +23.8647] | +16.2729 [+10.6294, +22.1210] | +1.6114 [+1.1182, +2.1127] | +13.2402 [+9.1990, +17.6121] |
| confirm/vidstg | +24.3754 [+9.2308, +41.7397] | +22.5864 [+8.5625, +38.6946] | +1.7890 [+0.4448, +3.4180] | +17.2949 [+5.5459, +33.1138] |
| confirm/hc2 | +13.1648 [+4.5320, +24.3654] | +12.2274 [+3.8268, +23.1142] | +0.9373 [+0.4258, +1.4511] | +10.0621 [+3.9362, +17.9041] |

Cellwise GT-time−old8=(grid−old8)+(GT-time−grid), with the same identity for new8. Grid enumerates every i<j endpoint pair on the original MERGED observed frame grid, using [frame_i,frame_j+1). It allows cross-offset endpoints that the original two-offset envelope MAP need not visit. It does not invent unseen intermediate endpoints, change sampling or send GT time into the decoder. Official dense interpolation, no spatial extrapolation and full annotated GT span in the denominator remain literal.

| Panel / dataset | Full-flow grid oracle (%) | Full-flow GT-time (%) | GT-time−grid (pp) |
|---|---:|---:|---:|
| search/vidstg | +33.4788 [+24.6983, +42.1851] | +34.7422 [+25.6260, +43.7910] | +1.2634 [+0.7111, +1.8746] |
| search/hc2 | +51.9812 [+45.9342, +57.8574] | +53.3610 [+47.2018, +59.3026] | +1.3798 [+1.0244, +1.7787] |
| confirm/vidstg | +54.4403 [+42.1901, +66.0452] | +55.6330 [+43.1619, +67.3234] | +1.1927 [+0.4206, +2.1544] |
| confirm/hc2 | +43.4069 [+30.3713, +55.5333] | +44.4055 [+31.0727, +56.7562] | +0.9986 [+0.6069, +1.4174] |

## Candidate eviction versus ranking damage

| Panel / dataset | Harmed corrupt expert cells | All new candidates worse than old A | A-quality candidate exists but scorer harms | Old A interval still in new pool | Higher critic score yet harms |
|---|---:|---:|---:|---:|---:|
| search/vidstg | 36 | 12 | 24 | 5/80 | 22 |
| search/hc2 | 40 | 12 | 28 | 5/80 | 30 |
| confirm/vidstg | 16 | 1 | 15 | 5/40 | 15 |
| confirm/hc2 | 28 | 13 | 15 | 6/40 | 17 |

Retaining central native does NOT retain the old critic-selected A interval. A forced-harm cell has new-pool oracle below old A, so no scorer over that new pool can preserve old performance there. A preservable-harm cell still has an A-quality candidate but the critic chooses worse. These are distinct limitations; negative actual results cannot be assigned entirely to the scorer. Counts are per expert arrival, not independent sources.


## Clean, ordering and negative outcomes

| Panel / dataset | Clean all actual gain (pp) | Corrupt expert order1 / order2 gain (pp) | Improved / harmed / unchanged expert cells | >5pp harm |
|---|---:|---:|---:|---:|
| search/vidstg | -0.8917 [-3.2466, +0.5294] | -0.6063/-7.5987 | 24/36/20 | 13 |
| search/hc2 | -1.3803 [-4.0194, +0.6710] | -1.8416/-13.8552 | 35/40/5 | 23 |
| confirm/vidstg | -0.6359 [-3.1599, +1.0313] | -6.5622/+2.0891 | 21/16/3 | 5 |
| confirm/hc2 | -1.5706 [-3.9603, +0.0402] | -3.2001/-8.2798 | 9/28/3 | 18 |

Gross gain/loss, strict .3/.5 correctness transitions, all clean/expert/nonexpert means, cell means, source influence, leave-one-source-out ranges and positive/negative anonymous cases are retained in ROWS/SUMMARY/CASES. Expert subsets have only7–16 distinct sources and differing membership across orders. Many correlated diagnostics are reported without a multiplicity-adjusted significance claim.

## Root interpretation and next variable

**The replacement allocation is not qualified for integration; retain A. The existing merged endpoint grid is not the dominant observed coverage limitation.** On corrupt expert arrivals, the grid explains90.05%,90.99%,92.66% and92.88% of the old eight-candidate coverage gap (development Vid/HC, confirmation Vid/HC), as ratios of source-macro means. Residual GT-time minus grid is1.096,1.611,1.789 and.937pp. All residuals remain positive; this is not proof that endpoint precision can never help. Full-flow residuals are1.263,1.380,1.193 and.999pp, much smaller than the corresponding time ideal headroom. The grid is an offline conditional oracle, not an unlabelled detector.
The allocation genuinely spreads center and duration and retains eight unique intervals, but has not demonstrated reliable oracle improvement. Corrupt expert new-minus-old oracle is−.471pp[−5.487,+3.962] on Vid development,+3.033[−1.007,+8.140] on HC development,+5.292[−.769,+13.652] on Vid confirmation and+2.165[−4.625,+10.653] on HC confirmation. Every paired interval crosses zero. Confirmation positive means should be preserved as opportunity, not elevated to a stable generator improvement. Both confirmation leave-one-source-out oracle means include positive values, but HC can turn negative when its most influential source is omitted.
Actual selection does not realize those opportunity means. Full-corruption new readout minus A is Vid development−1.026pp[−3.277,+.338],HC development−1.962[−4.537,−.060],Vid confirmation−.559[−2.512,+.639] andHC confirmation−1.435[−2.982,−.268]. On expert confirmation it is−2.237pp[−9.767,+2.467] and−6.768[−12.056,−2.492]. Both HC order means are negative; Vid confirmation reverses across orders. Clean full-stream means are also negative, so this is not a corruption-specific success.
There are TWO demonstrated failure paths. Native retention is weaker than retention of the old selected A interval: the new pool retains that interval in only5/80,5/80,5/40 and6/40 corrupt expert arrivals. Among36,40,16 and28 harmed arrivals,12,12,1 and13 have every new candidate below old A (support eviction). The remaining24,28,15 and15 still have an A-quality candidate but the critic selects a worse one (ranking failure). Higher critic scores nevertheless harm in22,30,15 and17 arrivals. These counts are not source-independent hypothesis tests and must not be collapsed into a universal scorer-only cause.
Preserve concrete negative and positive cases. Vid confirmation source36/order1/occlusion has identical old/new oracle55.4094% but actual55.4094→20.9652%; the scorer creates harm despite sufficient support. HC confirmation source43/order2/exposure changes old36.2993% to0%, while new oracle31.3400% is already below old A: candidate removal and erroneous ranking both matter. Conversely Vid confirmation source34/order1/frame-freeze improves19.1957→28.7055%, andHC source34/order1/motion-blur improves50.7093→53.8925%, with further unselected oracle opportunity. All old/new intervals are fixed before GT and these cases do not define online gates.
**Recommended next principal variable, if separately authorized: temporal localization-quality selection on a fixed sealed support.** The much larger merged-grid opportunity gives no reason to prioritize a finer endpoint grid now. The present replacement is not a validated candidate generator and must not be deployed. A later selector contrast should keep its support fixed and account for support-eviction cases, rather than claim that a better scorer can repair absent candidates. No new scorer, candidate variant, observation, loss, parameter update or full-query queue is executed in this task. Further candidate design remains necessary if selector improvement is bounded by the evicted support; this report does not select a GT-dependent fallback or dataset-specific winner.

## Actual incremental cost and audit

Generation CPU process wall 9.856s; scoring wall 73.705s; independent root audit wall 13.429s. These include loading/I/O and are not GPU kernel time or deployment latency. New model/expert/backbone/suffix/backward calls all0. Exhaustive grid pairs scored: 5,085,684. Literal official scalar checks: 27,648; maximum dense discrepancy 2e-15.

All288 current native intervals match cached source native; every old critic score/choice reproduces exactly; every new pool has8 distinct intervals,0stratum fallback slots. 5633 immutable input files are verified. Both historical A scalar outcomes and old eight-candidate oracle reproduce the preceding report. New global seal precedes GT exposure and all metric computation. Startup import-path failure before preparation was repaired and preserved; no prediction, label read or scientific rule changed in that repair.

Only this fixed allocation was tested. A large grid oracle is an interface-capacity result, not a proof that an unlabelled generator or critic can find the best interval. Temporal headroom is conditional on fixed A boxes, cannot be added to the spatial conditional upper bound, and does not establish temporal as the only overall bottleneck. The production registry and every old paused queue remain unchanged. Code, protocol, all anonymous outcomes and figures are public-exported; media/captions/annotations/GT coordinates/weights/raw tensor caches remain excluded.
