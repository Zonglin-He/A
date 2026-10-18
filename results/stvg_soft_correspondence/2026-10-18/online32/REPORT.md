# Four-arm actual online dense correspondence feedback32

Sixteen unique parent queries/domain, selected by the first eligible original P1 order1 parent after excluding recent rollback32/P6_64/original tuning parents. Historically full-stream exposed; these are new to the recent mechanism panel but not unseen confirmation. Four Source-initialized independent branches carry their own LN state in this newly filtered16-query stream; QR and Adam reset at each query. It does not replay the original full online history.

A is unchanged sparse OPD; B adds raw frozen correspondence; C adds correspondence rankings with interpolation-derived weight strength; D adds continuous box-interpolation feedback. Dense term and sparse term are separately frame-averaged with coefficient one. Final Native time, Source checkpoints and original domain sigma/tau/lr/steps/writeback/1792 parameters stay fixed. No new DINO or training. Sparse-plus-dense uses two separately normalized losses; it does not match the total gradient budget of A. Actual gradients and Adam movements are measured. New aliases H001/V001 refer to different parents from older panels; no old identity labels are joined.

## Complete metrics

| Domain | Arm | State | vIoU % | tIoU % | sIoU % | R03 % | R05 % |
|---|---|---|---:|---:|---:|---:|---:|
| hc2 | A_sparse | Source | 10.603255 | 24.378113 | 54.970017 | 6.250000 | 0.000000 |
| hc2 | A_sparse | Before | 10.824577 | 24.378113 | 56.464838 | 6.250000 | 0.000000 |
| hc2 | A_sparse | After | 12.517803 | 24.378113 | 58.614313 | 12.500000 | 0.000000 |
| hc2 | B_raw | Source | 10.603255 | 24.378113 | 54.970017 | 6.250000 | 0.000000 |
| hc2 | B_raw | Before | 11.064583 | 24.378113 | 57.017060 | 6.250000 | 0.000000 |
| hc2 | B_raw | After | 12.361196 | 24.378113 | 54.372119 | 6.250000 | 0.000000 |
| hc2 | C_rank | Source | 10.603255 | 24.378113 | 54.970017 | 6.250000 | 0.000000 |
| hc2 | C_rank | Before | 11.171091 | 24.378113 | 57.027228 | 6.250000 | 0.000000 |
| hc2 | C_rank | After | 12.100749 | 24.378113 | 52.725998 | 6.250000 | 0.000000 |
| hc2 | D_interpolation | Source | 10.603255 | 24.378113 | 54.970017 | 6.250000 | 0.000000 |
| hc2 | D_interpolation | Before | 11.286723 | 24.378113 | 57.090749 | 6.250000 | 0.000000 |
| hc2 | D_interpolation | After | 11.683539 | 24.378113 | 52.225437 | 12.500000 | 0.000000 |
| vidstg | A_sparse | Source | 8.015175 | 24.848254 | 38.561291 | 0.000000 | 0.000000 |
| vidstg | A_sparse | Before | 8.598536 | 24.848254 | 40.044261 | 6.250000 | 0.000000 |
| vidstg | A_sparse | After | 9.894375 | 24.848254 | 40.069848 | 12.500000 | 0.000000 |
| vidstg | B_raw | Source | 8.015175 | 24.848254 | 38.561291 | 0.000000 | 0.000000 |
| vidstg | B_raw | Before | 8.757630 | 24.848254 | 40.597368 | 6.250000 | 0.000000 |
| vidstg | B_raw | After | 10.037064 | 24.848254 | 38.203528 | 12.500000 | 0.000000 |
| vidstg | C_rank | Source | 8.015175 | 24.848254 | 38.561291 | 0.000000 | 0.000000 |
| vidstg | C_rank | Before | 8.850268 | 24.848254 | 40.867922 | 6.250000 | 0.000000 |
| vidstg | C_rank | After | 10.092851 | 24.848254 | 39.215732 | 12.500000 | 0.000000 |
| vidstg | D_interpolation | Source | 8.015175 | 24.848254 | 38.561291 | 0.000000 | 0.000000 |
| vidstg | D_interpolation | Before | 8.510947 | 24.848254 | 39.057201 | 6.250000 | 0.000000 |
| vidstg | D_interpolation | After | 9.356041 | 24.848254 | 32.952969 | 12.500000 | 0.000000 |

## Prespecified outcome and immediate-versus-inherited effects

The two-domain gate failed because HC C does not exceed A and its leave-one-scored-parent-out condition fails; the Vid mean/LOO/harm conditions pass. The stop follows this prespecified conjunction, not a rule that every interval crossing zero proves no effect. C−D mean gains and their uncertainty remain. No confirmation, direct-control run, temperature/encoder search or promotion is dispatched.

| Domain | C minus A total Δv pp | C minus A current Δv pp | C minus A inherited Δv pp |
|---|---:|---:|---:|
| hc2 | -0.417053 | -0.763567 | +0.346514 |
| vidstg | +0.198476 | -0.053256 | +0.251732 |

Vid C−A total benefit is accompanied by a negative current-query difference and a larger positive inherited-state difference. It cannot be described as a measured immediate dense-update improvement. HC has the same sign split but negative total difference. These are algebraic state-score contrasts under four different coupled histories, not causal shares or proof of long-stream transfer.

## Current and inherited changes

Total=After−Source, current=After−Before, inherited=Before−Source. This exact arithmetic does not identify causal cross-video transfer. One query/parent makes query macro equal parent macro. Parent bootstrap10000 draws is descriptive conditional on this coupled online sequence.

| Domain | Arm | Total pp [CI95] | Current pp [CI95] | Inherited pp [CI95] | Current harms below −5pp |
|---|---|---:|---:|---:|---:|
| hc2 | A_sparse | +1.914547 [+0.203535, +3.858413] | +1.693225 [+0.023003, +3.586397] | +0.221322 [+0.080830, +0.365549] | 0 |
| hc2 | B_raw | +1.757941 [+0.044332, +3.600547] | +1.296613 [-0.334541, +3.094986] | +0.461327 [+0.218614, +0.728901] | 1 |
| hc2 | C_rank | +1.497494 [-0.249559, +3.125482] | +0.929658 [-0.715582, +2.530053] | +0.567836 [+0.240596, +0.943344] | 1 |
| hc2 | D_interpolation | +1.080284 [-0.684662, +3.012430] | +0.396816 [-1.246565, +2.254970] | +0.683468 [+0.333615, +1.070862] | 1 |
| vidstg | A_sparse | +1.879200 [+0.625709, +3.352992] | +1.295839 [+0.404635, +2.326357] | +0.583361 [+0.141658, +1.146161] | 0 |
| vidstg | B_raw | +2.021889 [+0.703654, +3.530697] | +1.279434 [+0.444000, +2.225330] | +0.742455 [+0.156629, +1.481231] | 0 |
| vidstg | C_rank | +2.077676 [+0.772976, +3.568599] | +1.242583 [+0.447663, +2.171803] | +0.835093 [+0.181057, +1.670108] | 0 |
| vidstg | D_interpolation | +1.340866 [+0.347487, +2.630532] | +0.845094 [-0.237060, +2.331871] | +0.495772 [-0.135772, +1.335888] | 0 |

## Effective fit counts

No admitted reference is distinct from actual zero updates. Queries with references may still have all-equal sparse feedback; dense feedback can be informative in that case. Fixed maximum steps/samples are shared, but actual gradient-call counts are reported per arm.

| Domain | Queries without references | Arm | Queries with updates | Queries with zero updates | Reference present but no update | Parameter updates |
|---|---:|---|---:|---:|---:|---:|
| hc2 | 1 | A_sparse | 15 | 1 | 0 | 600 |
| hc2 | 1 | B_raw | 15 | 1 | 0 | 600 |
| hc2 | 1 | C_rank | 15 | 1 | 0 | 600 |
| hc2 | 1 | D_interpolation | 15 | 1 | 0 | 600 |
| vidstg | 4 | A_sparse | 11 | 5 | 1 | 110 |
| vidstg | 4 | B_raw | 12 | 4 | 0 | 120 |
| vidstg | 4 | C_rank | 12 | 4 | 0 | 120 |
| vidstg | 4 | D_interpolation | 12 | 4 | 0 | 120 |

## Same-panel paired comparisons

| Domain | After comparison | Δv pp [CI95] | Positive / negative / neutral | Leave-one-scored-parent-out minimum pp |
|---|---|---:|---:|---:|
| hc2 | B_raw_minus_A_sparse | -0.156606 [-0.570709, +0.234332] | 4/7/5 | -0.287677 |
| hc2 | B_raw_minus_D_interpolation | +0.677657 [-0.058729, +1.606243] | 6/5/5 | +0.354985 |
| hc2 | C_rank_minus_A_sparse | -0.417053 [-1.185522, +0.275074] | 4/7/5 | -0.596214 |
| hc2 | C_rank_minus_D_interpolation | +0.417210 [-0.446933, +1.373846] | 7/4/5 | +0.088527 |
| hc2 | D_interpolation_minus_A_sparse | -0.834263 [-1.823028, +0.010369] | 5/6/5 | -0.957614 |
| vidstg | B_raw_minus_A_sparse | +0.142689 [-0.114227, +0.483239] | 5/4/7 | +0.001627 |
| vidstg | B_raw_minus_D_interpolation | +0.681023 [+0.006371, +1.498472] | 7/2/7 | +0.408703 |
| vidstg | C_rank_minus_A_sparse | +0.198476 [-0.192717, +0.652644] | 4/5/7 | +0.022879 |
| vidstg | C_rank_minus_D_interpolation | +0.736810 [-0.018821, +1.614967] | 6/3/7 | +0.429954 |
| vidstg | D_interpolation_minus_A_sparse | -0.538333 [-1.108381, -0.069287] | 2/7/7 | -0.624641 |

Leave-one-out drops one saved parent score from an average; it does not rerun the online stream with that update removed. PAIRED_DECOMPOSITION.json additionally separates C−A/C−D current and inherited effects, and matched observed/unobserved/far actual-decoder current-IoU changes with signed parents and descriptive intervals.

## Actual decoder support and unknowns

Official interpolated tube scores above are distinct from actual decoder-frame geometry below. Known annotation only; missing support is unknown, not zero. All four planned DINO slots, including rejected slots, are excluded from unobserved. Far uses physical distance≥25% input span from any planned slot.

| Domain | Support | Parents with known frames / 16 | Known frames | Arm | Source s % | Before s % | After s % |
|---|---|---:|---:|---|---:|---:|---:|
| hc2 | observed | 11/16 | 33 | A_sparse | 44.331645 | 45.513951 | 51.704497 |
| hc2 | observed | 11/16 | 33 | B_raw | 44.331645 | 46.634995 | 51.544083 |
| hc2 | observed | 11/16 | 33 | C_rank | 44.331645 | 46.952095 | 50.892826 |
| hc2 | observed | 11/16 | 33 | D_interpolation | 44.331645 | 47.515277 | 46.518216 |
| hc2 | unobserved | 16/16 | 382 | A_sparse | 55.367011 | 56.818186 | 58.815331 |
| hc2 | unobserved | 16/16 | 382 | B_raw | 55.367011 | 57.288834 | 54.678837 |
| hc2 | unobserved | 16/16 | 382 | C_rank | 55.367011 | 57.235965 | 52.827667 |
| hc2 | unobserved | 16/16 | 382 | D_interpolation | 55.367011 | 57.256969 | 52.607374 |
| hc2 | far | 11/16 | 148 | A_sparse | 58.505629 | 60.088331 | 58.083989 |
| hc2 | far | 11/16 | 148 | B_raw | 58.505629 | 61.307827 | 52.309201 |
| hc2 | far | 11/16 | 148 | C_rank | 58.505629 | 60.933546 | 49.645844 |
| hc2 | far | 11/16 | 148 | D_interpolation | 58.505629 | 61.172471 | 49.435407 |
| vidstg | observed | 9/16 | 30 | A_sparse | 33.603999 | 35.957990 | 45.504020 |
| vidstg | observed | 9/16 | 30 | B_raw | 33.603999 | 36.479425 | 45.593227 |
| vidstg | observed | 9/16 | 30 | C_rank | 33.603999 | 36.780429 | 45.281541 |
| vidstg | observed | 9/16 | 30 | D_interpolation | 33.603999 | 34.927397 | 43.364649 |
| vidstg | unobserved | 16/16 | 831 | A_sparse | 39.475026 | 40.972922 | 40.150271 |
| vidstg | unobserved | 16/16 | 831 | B_raw | 39.475026 | 41.450295 | 38.358631 |
| vidstg | unobserved | 16/16 | 831 | C_rank | 39.475026 | 41.714061 | 39.653465 |
| vidstg | unobserved | 16/16 | 831 | D_interpolation | 39.475026 | 39.736660 | 32.562792 |
| vidstg | far | 10/16 | 342 | A_sparse | 37.509657 | 37.839144 | 35.379625 |
| vidstg | far | 10/16 | 342 | B_raw | 37.509657 | 38.170900 | 31.854714 |
| vidstg | far | 10/16 | 342 | C_rank | 37.509657 | 38.250153 | 32.735390 |
| vidstg | far | 10/16 | 342 | D_interpolation | 37.509657 | 36.424539 | 24.137789 |

## Strength, actual updates and cost

C uses sorted interpolation weights calculated at C's own current Student and samples; after branches diverge these are not D's actual branch weights. Ties average assigned weights and can lower nominal strength; flat rewards are uniform. Equal weight concentration does not guarantee equal gradient/Adam movement. ROUND_DIAGNOSTICS.json retains every round's ESS, actual parameter gradient/step norms and actual decoder observed/unobserved movements; raw tensors/parameter vectors remain private. Dense-rank statistics use each branch's current samples and cannot by themselves establish actual tube gain. After updates, B/C/D candidate centers diverge: their rank deltas are not a strict same-candidate comparison of feedback quality; the earlier fixed-candidate69 audit provides that separate control.

| Domain | Arm | Mean fit sec | Updates/query | Sparse ESS | Dense ESS | Parameter step norm | Dense rank Δ pp |
|---|---|---:|---:|---:|---:|---:|---:|
| hc2 | A_sparse | 2.002474 | 37.500000 | 27.896138 | unknown | 0.167427 | unknown |
| hc2 | B_raw | 2.026429 | 37.500000 | 27.839024 | 31.778712 | 0.185548 | -0.002154 |
| hc2 | C_rank | 2.044480 | 37.500000 | 28.262921 | 29.608377 | 0.202257 | 0.002906 |
| hc2 | D_interpolation | 2.043219 | 37.500000 | 27.613177 | 28.747139 | 0.214960 | -0.009429 |
| vidstg | A_sparse | 0.515585 | 6.875000 | 30.432036 | unknown | 0.703779 | unknown |
| vidstg | B_raw | 0.544893 | 7.500000 | 30.478155 | 31.826535 | 0.746318 | 0.191816 |
| vidstg | C_rank | 0.551488 | 7.500000 | 30.470292 | 30.787137 | 0.756035 | 0.408758 |
| vidstg | D_interpolation | 0.543026 | 7.500000 | 30.345289 | 30.730927 | 0.805418 | -0.277383 |

## Engineering evaluation and actual pixels

Before any current GT scoring, the evaluator state-difference comparison was corrected to preserve float32 update arithmetic, and saved seed/antithetic/sigma/parameter-name audits were added. EVALUATION_LOCK002 binds the final scorer; original evaluation lock and correction description are retained privately. No model output, parameter state, gate or prediction was changed for that correction. Six actual private panels show new H001/V001 four-slot Source/Before/After readouts (72 RGB cells), with original frame and DINO pixel hashes exact. This is engineering inspection only, no GT overlay and no new identity/semantic labels. V001 has no admitted reference and its no-update outputs remain, rather than being dropped. No older same-name case identity label is reused. Source checkpoint basenames and hashes and frozen model state hashes are in ACTUAL_PROTOCOL.json.

GPU lease 360.521487s includes Source loading/capture, transfers, verification and fitting, not pure kernel or first-download deployment time. 32 frozen Source captures,128 fit invocations including no-op sources, actual parameter updates 2870,0 new DINO calls,0 training. No-fit sources stay in all denominators. Full two-decoder outputs replayed exactly, analytic likelihood/Adam/state arithmetic independently checked; decoder Jacobian correctness is not independently proved by this audit.

## Resource enforcement and prelocked decision

Actual enforcement: systemd MemoryHigh16GiB/MemoryMax20GiB/SwapMax1GiB is automatic; stage resource_guard checks host available6GiB and disk free30GiB. GPU free4GiB was checked manually by root at progress points, not by an automatic device-floor gate in this runner. The shared GPU lease serializes cooperating jobs; it does not itself enforce physical GPU memory admission. No peak GPU usage is asserted.

```json
{
  "status": "stop_this_dense_interface",
  "groups": {
    "hc2": {
      "conditions": {
        "C_beats_A_and_D": false,
        "leave_one_out_positive": false,
        "no_large_harm_count_increase": true
      },
      "passed": false
    },
    "vidstg": {
      "conditions": {
        "C_beats_A_and_D": true,
        "leave_one_out_positive": true,
        "no_large_harm_count_increase": true
      },
      "passed": true
    }
  },
  "passed": false,
  "automatic_dispatch": false,
  "no_temperature_or_encoder_search": true,
  "production_unchanged": true
}
```

Additional correspondence evidence requires C>A and C>D in both domains, positive leave-one-scored-parent-out means, and C current-harm count≤A+1. No temperature/encoder sweep or automatic follow-up/production promotion occurs. Full signed/neutral/negative/unknown, common event-length/motion/area/query strata and state decomposition remain in RESULTS.json. Offline GT strata do not route online evidence. This filtered, previously exposed finite panel cannot establish original full-stream benefit or unseen confirmation. Actual pixel checks are engineering only; identity correctness is not inferred from IoU and no new semantic labels are created.
