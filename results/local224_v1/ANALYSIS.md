# Stable local224: completed fixed HC64 analysis

Full evaluation is saved and paused by the user: HC1902 scored, Vid6088/10303 preserved. Local trial is completed; not adopted. All data are repeatedly exposed development.

The main arm worsens Global6: tIoU -2.8476pp, 95% CI [-4.2146,-1.4851]; full vIoU -1.1929pp [-1.9711,-0.4430]. Fixed-space vIoU -1.2879pp [-2.0919,-0.5201]. One event is rescued across tIoU=.5, five previously correct events are harmed; five queries lose >5pp full vIoU.

## What actually failed

1. The proposed spatial zoom usually does not occur. Sixty of64 windows cover>=90% image area;55 are literally the whole frame, with identical shifted controls. Mean crop area is97.60%. Even before context expansion11 trajectory unions already cover both image dimensions;1.5 expansion makes55 full. This is a property of this full-timeline union rule, not evidence that all spatial guidance is impossible.

2. A smaller window is not automatically a larger target after the actual preprocessing. The code fixes HEIGHT224 and caps width at1.4*height. Effective vertical gain averages1.0014x, maximum1.0909x; effective horizontal gain averages1.0090x, maximum1.1734x. No2x detail-scale replacement is exercised. Merely cutting horizontally often reduces input width rather than increasing target pixels. Reciprocal crop width in the raw report is geometric extent, not effective pixel magnification.

3. Most accuracy loss exists in the same temporal interface without cropping. G+whole224 offset0 last-layer fusion already loses2.7490pp tIoU versus G. Cropping adds -0.0986pp [-0.3936,+0.1623] and -0.0059pp full vIoU [-0.1372,+0.1486], with1 improvement/3 declines/60 ties. This does not isolate offset omission from last-layer weighting and changing two-offset hull to a one-offset posterior. Do not call the entire -2.85pp a spatial-crop failure.

4. Correct location has no demonstrated extra time benefit: true minus shifted tIoU -0.0471pp [-0.4819,+0.3626]; full vIoU +0.0674pp [-0.1888,+0.3370]. Fifty-five controls are identical by geometry, which reduces this panel's ability to test location, and the nine changed-window cases do not provide a consistent gain.

## All nine non-full windows

| Query | Crop area | tIoU true-full pp | full vIoU true-full pp | tIoU true-shift pp | Start error change s | End error change s |
|---|---:|---:|---:|---:|---:|---:|
| H006 | 0.9167 | -6.8514 | -1.6615 | -8.4424 | +1.8711 | +0.2911 |
| H018 | 0.9239 | -2.0702 | -0.1950 | -2.0702 | +0.0000 | +0.2995 |
| H020 | 0.5473 | +5.1724 | +3.7885 | +2.7586 | -0.3200 | -0.2800 |
| H038 | 0.7844 | -2.5641 | -2.3080 | -4.8718 | +0.0000 | +0.3333 |
| H045 | 0.8833 | +0.0000 | +0.0000 | +6.8182 | +0.0000 | +0.0000 |
| H047 | 0.9187 | +0.0000 | +0.0000 | +0.8275 | +0.0000 | +0.0000 |
| H056 | 0.9047 | +0.0000 | +0.0000 | +5.2632 | +0.0000 | +0.0000 |
| H058 | 0.9239 | +0.0000 | +0.0000 | -3.2967 | -0.5990 | +0.5990 |
| H064 | 0.6625 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -1.8800 |

H020 is a real positive: tIoU +5.1724pp and full vIoU +3.7885pp over the full224 fusion control; both boundary errors decrease. H006 is a negative: tIoU -6.8514pp and start error +1.8711s. H018/H038 also worsen; keeping only H020 would misrepresent the panel. No crop adds a new .5-threshold rescue over the matched whole224 fusion control.

## Actual cost and validity

The changed-input GPU experiment ran128 local views plus one whole224 engineering view, no new teacher/448/updates. Local network wall averages1.8005s/query; original RGB decode averages0.9767s. Crop+network+posterior+shared decode+parse is2.8162s, separately from the mean0.0163s readout for all four evaluated arms, output IO and one-time model/parser load6.6988/.9617s. Two-view experimental query wall averages4.9272s; whole trial wall340.2910s includes setup/control/validation. Shared global224/eight-teacher/spatial capture is reused and excluded. No fresh matched448 execution occurred, so no controlled448 speedup is claimed.

Engineering whole224 logits and single-view MBR recover bitwise. The55 whole-image windows also recover old logits bitwise; all55 true/shifted identical windows agree. Independent128 posterior checks and256 brute-force utility checks pass (max2.22e-16);768 official/dense metric checks pass. The434 full frozen sources and6088 paused full predictions/receipts remain unchanged.

## Decision

Do not deploy this fixed full-timeline-union crop formulation; stop without scanning context, offset or fusion weights. The main constraints to address before a further design are actual pixel-scale gain and preservation of temporal evidence aggregation. This result does not test a genuinely2x zoomed stable crop and does not refute the broader spatial-computation-allocation route. Full and the deferred63-query448 supplement remain paused/waiting; no automatic full resume.
