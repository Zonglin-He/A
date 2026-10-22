# Deep-supported temporal MBR: fixed case-driven recheck

**The long-event Source criterion still fails. No full launch or resume.**
This cached-only experiment changes the temporal decision set, not the posterior,
spatial branch, model, precision, input sampling or eight expert observations.

## Failure mechanism and implemented readout

The previous case audit showed shallow averaging can withdraw late corrections.
But v6 demonstrated that extrapolating those corrections harms short events.
We therefore preserve Global6's original six-layer posterior and expected-IoU MBR
objective, restricting the start/end choices to the minimum/maximum values of the
individual layer4/5/6 MBR boundaries. The final Source readout is always feasible.
No posterior penalty, event-length switch, constraint margin or fitted weight is used.
A synthetic positive control confirms early layers cannot reverse a shared final
boundary. Source/G6 compatibility and logit-gauge tests also pass.

The three modes were fixed before predictions: the primary late bracket, the
first3-layer bracket countercontrol and simple last3-layer posterior mean. No depth,
width, weight or temperature scan was performed. Layer count3 matches the minimum
history already examined in the preceding two-step continuation, not a GT-selected
per-query choice. Spatial DP remains the same and is executed on each readout interval;
fixed-spatial-trajectory vIoU is also reported to isolate temporal changes.

| Readout | Short Δt vs Source pp | Long Δt vs Source pp | Overall Δt vs Source pp | Full Δv vs Source pp |
|---|---:|---:|---:|---:|
| [early_bracket](tune/early_bracket.json) | +2.4332 | -0.1926 | +0.5459 | +1.1548 |
| [last3_mean](tune/last3_mean.json) | +0.0710 | +0.1136 | +0.1016 | +0.7876 |
| [late_bracket](tune/late_bracket.json) | +0.1393 | +0.2365 | +0.2092 | +0.9042 |

The primary has positive Source increments for both groups on128 tuning parents,
but small values(+.1393 short/+.2365 long). It fails the original extra tune+.5pp
and Global6-preservation gates; those failed gate fields remain intact. Before any
v7 recheck, a [transparent Source-objective extension](Source_qualification_extension.json)
freezes the same sole primary candidate to answer the user's explicit Source goal.
It does not declare the original protocol passed or select a fallback control.
The same35 previously exposed parents are used; prior baseline cases also informed
the mechanism. This is iterative development, not an independent test.

## Frozen35-parent recheck

| GT-duration group | Parents | Source tIoU% | Candidate tIoU% | ΔSource pp | Paired95% CI pp |
|---|---:|---:|---:|---:|---|
| short | 11 | 29.3042 | 30.6632 | +1.3590 | [-0.4267, +3.3791] |
| long | 24 | 52.1018 | 51.3035 | -0.7984 | [-2.4026, +0.7394] |

Overall tIoU is44.8165% versus Source44.9369%(−.1203pp), also−.0920pp versus
Global6. Full vIoU is28.9421% versus Source28.0863%, but that includes the existing
BoxFixed spatial benefit; it does not establish a temporal improvement.
Compared with Global6, full vIoU falls.0804pp, and the fixed-space temporal control
also declines. The number of full-vIoU losses>5pp versus Source remains5.

The following same-case changes explain why selected-case repair is insufficient:

| Case | Source tIoU% | Global6 tIoU% | Late-bracket tIoU% | Effect versus Global6 |
|---|---:|---:|---:|---:|
| P131 |87.023|69.466|75.573|+6.107pp partial repair|
| P141 |39.583|33.249|34.635|+1.387pp partial repair|
| P132 |88.293|100.000|92.195|−7.805pp, removes a true ensemble correction|
| P152 |22.912|26.730|22.912|−3.819pp, returns to Source error|
| P158 |69.959|81.481|77.778|−3.704pp, reduces a genuine correction|

Deep agreement is not a correctness certificate. The guard fixes some effects of
premature averaging but also excludes ensemble decisions that are better than the
individual late readouts. Some wrong original decisions and semantic failures are
not fixed by either rule. These findings weaken both tested heuristics; they do not
prove all temporal fusion or all cache-based readout mechanisms are impossible.

[All signed recheck metrics](validation/late_bracket.json), [strict Source outcome](outcome.json),
[protocol](protocol.json), [locked3-mode schedule](schedule.json),
[engineering/hash verification](verification.json). The failed original gates and
failed Source qualification are separately retained. No fallback after recheck,
no full score used for selection, no runtime fingerprint exemption. 50 CPU tests
passed;40 paused core files and434 provider files remain exact;1059 cached trial
prediction records were sealed then scored with official/dense agreement.

The old time-modified full remains paused at338/1902. Existing original complete
full results remain preserved. The CLI/default is not upgraded; a failed profile
will not be resumed as an exploratory run. No new GPU/model/expert/teacher calls.
