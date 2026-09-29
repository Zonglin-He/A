# Phase A: matched ablations of the frozen J0 recipe

All three controls completed in one batch on the same16previously exposed VidSTG sources, five prelocked J0.1 orders, clean and five5%seed0conditions. Final method unchanged. No winner selected or configuration retuned.

| Method | Whole vIoU % | Whole Δv vsFrozen pp | Future Δs vsFast pp | Future Δv vsFast pp | Whole source >5pp v-harm count/16 | Future source >5pp v-harm count/12 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | +15.845411 ± 0.000000 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 |
| Fast-only | +17.341651 ± 0.863637 | +1.496239 ± 0.863637 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +0.200000 ± 0.447214 | +0.000000 ± 0.000000 |
| Slow-only / R-OPD | +15.866868 ± 0.015196 | +0.021456 ± 0.015196 | +0.094390 ± 0.071792 | +0.025505 ± 0.020209 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 |
| Final (frozen) | +17.363092 ± 0.859943 | +1.517681 ± 0.859943 | +0.094390 ± 0.071792 | +0.025505 ± 0.020209 | +0.200000 ± 0.447214 | +0.000000 ± 0.000000 |
| Random-Rank | +17.340186 ± 0.859934 | +1.494774 ± 0.859934 | -0.009502 ± 0.047416 | -0.004011 ± 0.016776 | +0.200000 ± 0.447214 | +0.000000 ± 0.000000 |
| Off-Policy | +17.363531 ± 0.860644 | +1.518120 ± 0.860644 | +0.096569 ± 0.070254 | +0.026227 ± 0.019368 | +0.200000 ± 0.447214 | +0.000000 ± 0.000000 |
| Direct PL | +17.344484 ± 0.862892 | +1.499073 ± 0.862892 | +0.009635 ± 0.006329 | +0.003409 ± 0.002938 | +0.200000 ± 0.447214 | +0.000000 ± 0.000000 |

Values are equal-order mean ± sampleSD across5orders; orders share sources, not independent datasets. Source corruption mean averages the five conditions first. Harm is paired source-mean delta<−5pp; per-cell harm and sIoU harm are separately retained in SUMMARY/ACROSS_ORDERS. Nonexpert Fast equals Frozen.

## Paired controls versus Final

| Control | Whole control−Final Δv pp | Future control−Final Δv pp | Future Δv positive/negative orders |
|---|---:|---:|---|
| Random-Rank | -0.022906 ± 0.023157 | -0.029515 ± 0.033549 | 3/2 |
| Off-Policy | +0.000439 ± 0.001024 | +0.000722 ± 0.001270 | 4/1 |
| Direct PL | -0.018608 ± 0.015520 | -0.022095 ± 0.019660 | 5/0 |

## Every locked order

| Method/order | Whole Δv pp | Future Δs pp | Future Δv pp | Whole source v-harm | Future source v-harm |
|---|---:|---:|---:|---:|---:|
| Final (frozen) / order1 | +1.987420 | +0.177227 | +0.052204 | 0 | 0 |
| Final (frozen) / order2 | +2.224157 | -0.013591 | -0.003254 | 0 | 0 |
| Final (frozen) / order3 | +1.449875 | +0.067394 | +0.034033 | 0 | 0 |
| Final (frozen) / order4 | +1.863299 | +0.122259 | +0.022202 | 0 | 0 |
| Final (frozen) / order5 | +0.063653 | +0.118661 | +0.022337 | 1 | 0 |
| Random-Rank / order1 | +1.938896 | -0.038927 | -0.013957 | 0 | 0 |
| Random-Rank / order2 | +2.232465 | +0.014400 | +0.007145 | 0 | 0 |
| Random-Rank / order3 | +1.407004 | -0.078433 | -0.028608 | 0 | 0 |
| Random-Rank / order4 | +1.845067 | +0.023682 | +0.004228 | 0 | 0 |
| Random-Rank / order5 | +0.050439 | +0.031771 | +0.011138 | 1 | 0 |
| Off-Policy / order1 | +1.987803 | +0.180238 | +0.052881 | 0 | 0 |
| Off-Policy / order2 | +2.226224 | -0.006219 | -0.000486 | 0 | 0 |
| Off-Policy / order3 | +1.449381 | +0.066235 | +0.033382 | 0 | 0 |
| Off-Policy / order4 | +1.863899 | +0.125408 | +0.022917 | 0 | 0 |
| Off-Policy / order5 | +0.063290 | +0.117183 | +0.022440 | 1 | 0 |
| Direct PL / order1 | +1.950528 | +0.005659 | +0.001942 | 0 | 0 |
| Direct PL / order2 | +2.228103 | +0.002594 | +0.001694 | 0 | 0 |
| Direct PL / order3 | +1.438295 | +0.015813 | +0.008601 | 0 | 0 |
| Direct PL / order4 | +1.837126 | +0.007314 | +0.001923 | 0 | 0 |
| Direct PL / order5 | +0.041314 | +0.016796 | +0.002885 | 1 | 0 |

## Clean control

| Method | Whole Δv pp | Future Δs pp | Future Δv pp |
|---|---:|---:|---:|
| Frozen | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 |
| Fast-only | +1.439705 ± 0.481775 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 |
| Slow-only / R-OPD | +0.023936 ± 0.015737 | +0.091562 ± 0.072229 | +0.028776 ± 0.021239 |
| Final (frozen) | +1.463694 ± 0.480370 | +0.091562 ± 0.072229 | +0.028776 ± 0.021239 |
| Random-Rank | +1.438333 ± 0.472795 | -0.011429 ± 0.051008 | -0.003892 ± 0.019291 |
| Off-Policy | +1.463994 ± 0.481249 | +0.094098 ± 0.066732 | +0.029605 ± 0.019726 |
| Direct PL | +1.444061 ± 0.482135 | +0.009453 ± 0.010648 | +0.005714 ± 0.008476 |

## Actual interventions and interpretation

Random-Rank permutes rank assignment using a prelocked source/order hash and preserves the rank multiset and probability spectrum of its own current-policy support. Diverged states can lead to different later supports/rewards from Final; no claim that the complete trajectories retain equal update norms or q entropy.

Off-Policy generates current-sample probes from fixed source parameters, but fits the current persistent actor against those detached boxes. Temporal candidates remain current-policy.

Direct PL uses the same valid sparse Sa2VA masks,1792parameters,SGD.005 and pre-update output; mean frame loss is sum-coordinate L1+GIoU with coefficients1/1 as specified. No probes, norm matching or loss-scale tuning. This tests the fixed configuration, not all pseudo-label algorithms.

Existing Raw/Rank/Rank+Norm results remain in S1.1 with their original single ordinal order. They are historical, not five-order matched rows. Existing critic/probe qualifications and J0/J0.1 schedule evidence are reused without new GPU reruns.

Execution: 417.239GPU-process seconds, 306updates,2160spatial probes,0new experts/encoder captures. 1440state links/90resets, 548352SGD coordinates, 204rank KL and102PL losses, 1440dual metrics,12full spatial/6six-layer temporal reinsertions,3CPU tests and17076public scalar checks. Predictions sealed before re-reading16old labels.

Ablation outcomes constrain empirical claims, not frozen method selection. PhaseB remains the unchanged recipe; all same-source/condition/order dependence and historical exposure are retained.

## Empirical claim readout

Specialist preference is supported against this fixed random-assignment control: future Final +0.025505 pp versus Random −0.004011 pp. Direct PL yields +0.003409 pp, below Final in this fixed one-step/loss-scale comparison. **On-policy advantage is not established:** Off-Policy +0.026227 pp is essentially equal and slightly above Final (+0.000722 pp paired). Do not claim the on-policy support explains the measured benefit; keep the frozen implementation without selecting a replacement.
