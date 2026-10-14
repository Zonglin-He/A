# Spatial causal decomposition: fixed-reference amplitude diagnostic

Phase 1 measurements and independent readback are complete; the three-stage research program is not complete. Temporal and Spatial remain the intended dual-branch method. No component removal, CURRENT promotion, old queue recovery, or full-scale expansion.

The fixed cohort is 16 historically exposed development sources, eight per domain. HC04/V04 endpoints refer to Motion-Guided Instance OPD, not Adjacent Motion. References, positions, saved stochastic updates and 1792-parameter directions are fixed. Alpha=[0,1/4,1/2,1] scales the realized final parameter displacement, not learning rate or optimizer trajectories. Missing multipliers in the request were resolved by this prelocked finite default.

Three nonzero cases received 3 fresh source captures plus 9 additional full-model readouts, yielding 12 actual outputs. Ten locked no-update and three zero-displacement fits reuse exact source outputs. No new teacher call, fit, backward or optimizer step. Both endpoints reproduce sealed arrays exactly; 96 readouts are scored postseal with official and independent dense metrics (maximum difference 1.23e-15). Original-eight T is exactly reused; additional-eight T is unavailable.

## Fixed-direction amplitude results

All values below are full-Tube vIoU percentage-point differences from Frozen, not sparse anchor IoU.

| Case | 0.25 | 0.5 | 1 |
|---|---:|---:|---:|
| HC01 | +0.000000 | +0.000000 | +0.000000 |
| HC02 | +0.000000 | +0.000000 | +0.000000 |
| HC03 | +0.000000 | +0.000000 | +0.000000 |
| HC04 | -3.169574 | -5.915793 | -10.657945 |
| HC05 | +0.000000 | +0.000000 | +0.000000 |
| HC06 | +0.000000 | +0.000000 | +0.000000 |
| HC07 | +0.000000 | +0.000000 | +0.000000 |
| HC08 | +0.000000 | +0.000000 | +0.000000 |
| V01 | +0.000000 | +0.000000 | +0.000000 |
| V02 | -0.469960 | -1.154153 | -2.169210 |
| V03 | +0.000000 | +0.000000 | +0.000000 |
| V04 | +7.102443 | +9.343894 | +9.995520 |
| V05 | +0.000000 | +0.000000 | +0.000000 |
| V06 | +0.000000 | +0.000000 | +0.000000 |
| V07 | +0.000000 | +0.000000 | +0.000000 |
| V08 | +0.000000 | +0.000000 | +0.000000 |

Reducing displacement reduces HC04 harm and retains much of V04 gain. HC04 and V02 remain negative at every nonzero tested alpha. This supports amplification by update magnitude; it does not establish that smaller updates fix a wrong reference. No per-case GT-selected alpha is adopted.

## Candidate support and teacher selection

Use only GT-known positions shared by Source and every prelocked displayed candidate. Of 16 sources, 6 have no common support and remain unknown. Of the other 10, 9 have one common position and HC04 has two. Pool Oracle delta is positive in 4 and neutral in 6; because Source is included, a nonnegative maximum alone is not evidence of quality. Locked teacher delta: 1 positive, 2 negative, 7 neutral, 6 unknown. All 78 displayed candidate IDs are included; five other scored candidates remain separately marked as not displayed. Per-candidate pairwise supports differ and are not pooled into the common-support Oracle. Sparse IoU deltas and full-Tube vIoU deltas are never subtracted from one another.

HC04: same-source, same-position, same-config and same-seed historical offline true-reference IoU-OPD improves full-Tube vIoU by 6.2075 pp. The current selected reference worsens sparse common-support quality and the original update loses 10.6579 pp. This is a useful supervision positive control. Other cases can differ in observation support; the eight offline Oracle results are not uniformly matched reference-only causal contrasts, nor deployment results. No new Oracle fit.

## Full paired domain readouts

S delta is versus Frozen; TS delta is versus the exact reused Temporal-only baseline. Original and additional cohorts are never pooled as independent confirmation.

| Cohort | Domain | Branch | alpha | mean vIoU % | delta pp |
|---|---|---|---:|---:|---:|
| original8 | hc2 | S | 0 | 22.210455 | +0.000000 |
| original8 | hc2 | S | 0.25 | 21.418062 | -0.792393 |
| original8 | hc2 | S | 0.5 | 20.731507 | -1.478948 |
| original8 | hc2 | S | 1 | 19.545969 | -2.664486 |
| original8 | hc2 | TS | 0 | 31.286031 | +0.000000 |
| original8 | hc2 | TS | 0.25 | 30.462641 | -0.823390 |
| original8 | hc2 | TS | 0.5 | 29.682636 | -1.603395 |
| original8 | hc2 | TS | 1 | 28.305631 | -2.980399 |
| original8 | vidstg | S | 0 | 14.490136 | +0.000000 |
| original8 | vidstg | S | 0.25 | 16.148257 | +1.658121 |
| original8 | vidstg | S | 0.5 | 16.537571 | +2.047435 |
| original8 | vidstg | S | 1 | 16.446714 | +1.956578 |
| original8 | vidstg | TS | 0 | 22.441929 | +0.000000 |
| original8 | vidstg | TS | 0.25 | 24.119173 | +1.677244 |
| original8 | vidstg | TS | 0.5 | 24.273800 | +1.831871 |
| original8 | vidstg | TS | 1 | 23.838503 | +1.396574 |
| additional8 | hc2 | S | 0 | 14.376713 | +0.000000 |
| additional8 | hc2 | S | 0.25 | 14.376713 | +0.000000 |
| additional8 | hc2 | S | 0.5 | 14.376713 | +0.000000 |
| additional8 | hc2 | S | 1 | 14.376713 | +0.000000 |
| additional8 | vidstg | S | 0 | 3.193724 | +0.000000 |
| additional8 | vidstg | S | 0.25 | 3.193724 | +0.000000 |
| additional8 | vidstg | S | 0.5 | 3.193724 | +0.000000 |
| additional8 | vidstg | S | 1 | 3.193724 | +0.000000 |

## Remaining authorized stages

Phase 2 is a separate finite, prelocked short-window candidate/association experiment near the existing QA event, with unchanged student optimizer. Existing source caches contain no saved spatial feature maps; obtaining localized frozen features requires new capture with explicit frame/grid/resize binding. Short support and feature association require matched controls; a valid-track count is not a candidate-quality result. Phase 3 requires one fixed GT-free Spatial rule and verifiably unused sources (at least 32 per direction if available); it has not been launched. Final confirmation must include F/T/S/TS.

## Engineering and evidence boundaries

A concurrent agent added one required-pin constant after declaring source frozen. Both versions were retained; the byte-exact pinned original was restored. The active process had passed its old self-hash check before model loading; scientific functions, runtime, contract, inputs and measurements were unchanged. Root runtime already pinned the launcher. This is a disclosed engineering coordination error, not a scientific result. The own resource-guarded finite unit exited successfully; no observed OOM. The measurement wall time includes this finite run only, not complete TTA deployment cost. Full positive/negative/neutral/unknown anonymous scalars and source bindings accompany this report. No raw media, labels, query text, weights, parameter states or private caches are published.
