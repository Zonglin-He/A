# Temporal boundary pairing: strict short/long Source criterion

**The required two-group benefit was not achieved. Full evaluation remains paused.**
The user criterion is Source-relative tIoU gain strictly greater than zero in both
short(≤5s) and long(>5s) events, on tuning and fixed development recheck.
GT duration is used only offline. The same configuration applies to every input;
no GT duration, per-query best branch or teacher labels enter inference.
See [criterion locked before recheck](group_constraint.json).

This12-profile study reuses v4's163 sealed captures with no new GPU/model/expert runs.
It keeps128 tuning parents and the same35 previously exposed recheck parents.
The35 are a development recheck, not an independent test. Spatial code, weights,
observations and Source forward are unchanged. Zero pairing reproduces Global6 exactly.
44 CPU engineering tests passed;40 paused-run core files and434 provider source files
remain hash-identical. All private captures and paused full338 results remain intact.

## Mechanism and bounded experiment

Independent start/end scores can join the beginning of one event to the end of
another. We tested subtracting the soft probability mass of other boundaries inside
the interval, excluding a fixed endpoint neighborhood. This suppresses a bridge in
a synthetic two-event test, but genuine long events can also contain internal peaks.
Eight coarse profiles, two endpoint-radius refinements and two combinations with
already-measured MBR³ exhausted the predeclared12-profile budget.

| Configuration | Δt vs Source (pp) | Short Δt vs Source | Long Δt vs Source | Δv vs Source | Δv vs Global6 |
|---|---:|---:|---:|---:|---:|
| [Source_time](tune/Source_time.json) | +0.0000 | +0.0000 | +0.0000 | +0.7042 | -0.3525 |
| [base](tune/base.json) | +0.3455 | +1.4263 | -0.0774 | +1.0567 | +0.0000 |
| [negative100](tune/negative100.json) | +0.6831 | -2.1852 | +1.8054 | +0.9894 | -0.0673 |
| [pair050](tune/pair050.json) | -0.3504 | +2.6360 | -1.5189 | +0.8997 | -0.1570 |
| [pair050_radius0](tune/pair050_radius0.json) | -0.2641 | +2.3525 | -1.2880 | +0.9010 | -0.1557 |
| [pair050_radius0_risk300](tune/pair050_radius0_risk300.json) | -0.0814 | +3.2284 | -1.3765 | +1.1071 | +0.0504 |
| [pair050_radius2](tune/pair050_radius2.json) | -0.2965 | +2.2663 | -1.2994 | +0.8251 | -0.2316 |
| [pair050_radius2_risk300](tune/pair050_radius2_risk300.json) | +0.0939 | +3.2137 | -1.1269 | +1.1774 | +0.1207 |
| [pair100](tune/pair100.json) | -1.2818 | +3.6761 | -3.2219 | +0.4960 | -0.5607 |
| [pair200](tune/pair200.json) | -2.8863 | +4.6707 | -5.8434 | -0.2510 | -1.3077 |
| [pair400](tune/pair400.json) | -4.9241 | +4.1872 | -8.4893 | -1.5144 | -2.5711 |
| [risk300_control](tune/risk300_control.json) | +0.8088 | +1.6226 | +0.4903 | +1.2966 | +0.2399 |

Positive internal-peak penalties consistently improve short-event comparisons while
harming long events. Strength0.5→4 worsened the long-event Source delta from−1.519
to−8.489pp. Excluding0/2 cells and combining with MBR³ did not remove the long-event
loss. The negative-strength control reverses the tradeoff: long+1.805pp but short
−2.185pp. The peak mass therefore does not reliably distinguish a wrong bridge from
a valid multi-phase event. No such penalty is promoted.

The only admitted tuning configuration was the existing MBR³ control with no pairing
penalty. It evaluates expected IoU³ rather than expected IoU; this changes the output
decision utility, not the posterior or backbone. Before any v5 recheck we explicitly
amended pairing-only admission to test this sole qualifying control, in response to
the user's two-group objective. [Full admission and amendment](candidate_admission.json).
It is not described as a successful new pairing method. No fallback was chosen after
recheck and no default was changed.

## Frozen MBR³ recheck

| Panel | GT-duration group | Parents | Source tIoU% | Candidate tIoU% | ΔSource pp | Paired95% CI pp | t decline>5pp |
|---|---|---:|---:|---:|---:|---|---:|
| tune | short | 36 | 38.6741 | 40.2967 | +1.6226 | [-0.3714, +3.8840] | 6 |
| tune | long | 92 | 55.7014 | 56.1917 | +0.4903 | [-0.7807, +1.8787] | 11 |
| validation | short | 11 | 29.3042 | 30.1535 | +0.8493 | [-1.2786, +3.5002] | 0 |
| validation | long | 24 | 52.1018 | 51.4431 | -0.6587 | [-3.2499, +1.7727] | 4 |

The tuning two-group means pass, but both CIs include zero. The fixed35 recheck
fails long-event positivity(−0.6587pp) and overall Source t gain(−0.1847pp).
It also falls below Global6 on full vIoU(−0.2160pp) and unobserved sIoU(−0.1825pp).
Mean output length increases8.912→9.072s versus Global6. Baseline-good temporal
cases lose1.1084pp while baseline-bad cases gain0.8517pp, so the readout sometimes
rescues wrong modes but also shifts already-good boundaries. More decisive MBR utility
alone is insufficient. No per-query GT selection or further risk-power sweep follows.

[All signed35-query recheck results](validation/risk300_control.json),
[strict group aggregation and CIs](outcome.json),
[predeclared protocol](protocol.json), [locked schedule](coarse_schedule.json).
Raw scoring records retain their original gate fields; the later strict positivity
criterion is applied explicitly in outcome.json to both panels without changing any
prediction, sealed code or old score receipt.

All costs in these trials are cached CPU posterior/readout/DP and file-write costs;
they exclude Source/expert/cache construction and are not end-to-end deployment speed.
The earlier full stays paused at338/1902 with no scoring and no authorization to
resume. A future full requires a newly frozen Source-qualified configuration.
