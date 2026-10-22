# Source-primary temporal calibration: bounded v4

The selected configuration failed the fixed35-parent recheck. No full was resumed.
The earlier v3 exploratory full was an execution error: it had not beaten Source.
Its338 completed predictions and receipts are preserved under an explicit pause.

Only temporal output decisions changed. Source forward, inputs, precision, eight
ReferDINO observations and BoxFixed spatial implementation and parameters stayed fixed.
128 distinct prior tuning parents were retained; the remaining35 parents were fixed
before this study. V4 used163 sealed Source captures:137 content-bound complete reuse,
26 native captures, and zero new expert forwards. Predictions seal before independent
CPU labels/scoring. The35 recheck parents are now exposed development data.

The24 configurations test pooling, cross-offset agreement, start/end calibration,
label-free uncertainty conditioning, Source trust and MBR utility, followed by the
bounded measured-component combinations. See [protocol](protocol.json),
[locked schedule](coarse_schedule.json), [decisions](decisions.json) and
[selection](selection.json). All signed query-level values, including failures,
are retained in the linked reports below. No online GT choices are made.

| Configuration | Δt vs Source (pp) | Short Δt vs Source | Long Δt vs Source | Δv vs Source | Δv vs Global6 |
|---|---:|---:|---:|---:|---:|
| [Source_half](tune/Source_half.json) | +0.0118 | — | — | +0.8351 | -0.2216 |
| [Source_time](tune/Source_time.json) | +0.0000 | — | — | +0.7042 | -0.3525 |
| [Source_trust050](tune/Source_trust050.json) | -0.1755 | — | — | +0.7587 | -0.2980 |
| [Source_trust100](tune/Source_trust100.json) | -0.1727 | — | — | +0.7134 | -0.3433 |
| [adaptive000_risk300](tune/adaptive000_risk300.json) | +0.9198 | — | — | +1.2611 | +0.2044 |
| [adaptive050](tune/adaptive050.json) | +0.3447 | — | — | +0.9664 | -0.0903 |
| [adaptive050_risk300](tune/adaptive050_risk300.json) | +0.9591 | — | — | +1.3243 | +0.2676 |
| [adaptive100](tune/adaptive100.json) | +0.2430 | — | — | +0.8995 | -0.1572 |
| [adaptive_threshold000](tune/adaptive_threshold000.json) | +0.8174 | — | — | +1.1092 | +0.0525 |
| [adaptive_threshold050](tune/adaptive_threshold050.json) | +0.8974 | — | — | +1.1178 | +0.0611 |
| [agree100](tune/agree100.json) | +0.3174 | — | — | +1.0623 | +0.0056 |
| [agree200](tune/agree200.json) | +0.1827 | — | — | +1.0210 | -0.0357 |
| [agree400](tune/agree400.json) | -0.3185 | — | — | +0.7518 | -0.3049 |
| [base](tune/base.json) | +0.3455 | — | — | +1.0567 | +0.0000 |
| [disagree200](tune/disagree200.json) | +0.4599 | — | — | +1.0761 | +0.0194 |
| [nucleus090](tune/nucleus090.json) | -0.6049 | — | — | +0.7467 | -0.3100 |
| [nucleus099](tune/nucleus099.json) | +0.2118 | — | — | +1.0207 | -0.0360 |
| [old_common150](tune/old_common150.json) | +0.9503 | — | — | +1.0217 | -0.0350 |
| [old_risk200](tune/old_risk200.json) | +0.5928 | — | — | +1.2026 | +0.1459 |
| [old_start150](tune/old_start150.json) | +0.8324 | — | — | +1.0461 | -0.0106 |
| [pool_geo](tune/pool_geo.json) | +0.0411 | — | — | +0.9829 | -0.0738 |
| [pool_half](tune/pool_half.json) | +0.0816 | — | — | +0.9411 | -0.1156 |
| [risk150](tune/risk150.json) | +0.5412 | — | — | +1.1378 | +0.0811 |
| [risk300](tune/risk300.json) | +0.8088 | — | — | +1.2966 | +0.2399 |

The frozen entropy-conditioned start calibration plus MBR³ improved tuning tIoU
by+0.9591pp versus Source, but the fixed35 recheck gave Source44.9369%,
Global644.9085%, Candidate44.6655%: Source gain−0.2713pp.
Full vIoU was28.0863%,29.0225%,28.0894%, respectively, so Source spatial improvement
did not rescue the temporal failure. Candidate mean interval was9.945s versus
Global68.912s and Source8.632s. Start softening overexpanded already-good/short
intervals; increased MBR utility curvature did not make that calibration generalize.
These are measured output effects, not proof that the frozen backbone cannot improve.

[Fixed recheck: all signed metrics, duration strata, CIs and severe losses](validation/adaptive050_risk300.json).
[Outcome](outcome.json). No fallback after recheck; original main default remains unchanged.
