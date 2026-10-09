# P1 chart revision003: actual qualification and missing-only continuation

The human authorized the previously reviewed finite-native-logit endpoint
extension on 2026-10-09. The original exact-0/1 failure protocol and original
failure remain preserved and published. This is an explicit representation
revision, not a claim that the original chart qualification passed.

The fixed VidSTG configuration remains lr .03 / sigma .1 / tau .25 / 10 rounds /
LN writeback 1/8 / M32. HC2 retains .01 / .025 / .05 / 40 rounds / 1/16 / M32.
Native WHEN, Uniform4, the original admitted Top1 frozen DINO observation,
1792 joint parameters, antithetic on-policy Gaussian likelihood and detached
IoU, per-query residual/Adam reset, final-round output, source checkpoints,
target rosters and original runtime bytes are unchanged.

Only coordinates whose same-call float32 sigmoid rounds to 0 or 1 use their
finite native pre-sigmoid logit. Interior coordinates retain the exact original
logit(box). The native sigmoid box readout is unchanged. The raw/head binding
is checked on every decoder call; malformed or nonfinite values still fail.

The actual GPU qualification comprised six complete fits:

| Actual input | Validation | Actual GPU fit segments |
|---|---|---|
| First missing arrival1882 / query10220 | Repaired 10-round fit and complete repeated fit bitwise | .475896 s / .368758 s |
| Arrival1880 / query9914 | Original saved fit, unchanged no-feedback no-op, original/revised bitwise | .126462 s / .128899 s |
| Arrival1881 / query10224 | Original saved fit, informative 10-round interior fit, original/revised bitwise | .314417 s / .321045 s |

The original reproduced first eight complete rounds and ninth update match
210 tensors / 59,132 coordinates / 194 scalars bitwise. The two complete
repaired fits match 334 tensors / 69,266 coordinates / 330 scalars, including
the numerical trace. The two controls match 104 / 254 tensors and 24,232 /
67,526 coordinates respectively for all original result fields. One control
is explicitly a no-op, not an additional informative-gradient demonstration.

The real native-head-output VJP was checked across all ten informative rounds,
maximum error 1.192093e-7 under the predeclared 2e-6 threshold. Independent
CPU chart readback verified 20 before/after calls and three endpoint-coordinate
occurrences, maximum float64 interior formula discrepancy 4.121081e-7 and
native-sigmoid readout discrepancy 6.763823e-8, under explicitly saved float32
tolerances. Full Gaussian/IoU/mean-gradient/Adam/state arithmetic passed the
unchanged revision002 checks; root repeated the complete CPU audit in .011073 s.
Eight synthetic valid contracts and sixteen invalid-input/trace rejections
passed. These numerical checks do not establish GT effectiveness or a complete
independently reconstructed decoder Jacobian.

The old dead worker did not serialize the failed fit. Exact comparisons use
the retained real failure reproduction and newly repeated fits, never its
inaccessible dead memory. Qualification accepted zero new predictions and
made zero new DINO observations. The old 1,882-arrival VidSTG prefix was
rechecked as 547,699,038 bytes with original payload/receipt hashes intact.

Formal continuation accepted only the first missing arrival after its full
fit matched the actual qualification bitwise. Root readback verified payload
SHA/bytes, predecessor SHA, input binding, runtime/config, residual/Adam reset
and original LN writeback arithmetic. The initial root progress snapshot was
1,900 / 30,909 VidSTG arrivals; later dynamic stage STATUS is authoritative.
HC2's 10,446 sealed predictions are reused without another HC2 worker. Five
actual saved math dictionaries covering original, revision002 and revision003
receipts reproduced exactly on CPU without GT. Post-seal dispatch also retains
the pinned revision001 implementation for any saved revision001 receipt.

The finite controller continues the original VidSTG suffix and original
post-seal CPU stages. Both directional barriers and all 41,355 deployment
arrivals must precede any P1 GT evaluation. Full P1 root efficacy/dense/state
audits, actual report/case inspections, publication and P2-P6 remain pending.
EATA/media/Fisher and unrelated historical queues stay paused. Timed fit
segments exclude unmeasured load/capture/replay overhead; they are not total
recovery costs or full-run throughput. Private media, captions, annotations,
weights, predictions and gradient/Adam tensors are excluded from publication.
