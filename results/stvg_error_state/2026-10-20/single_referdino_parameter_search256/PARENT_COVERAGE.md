# Parent coverage of parameter configurations

Every denominator is all128 parents per target dataset. Changes are complete vIoU relative to the same initialization E_gamma+Strong. Any-positive uses1e−10pp numeric tolerance; meaningful improvement/decline is at least0.1pp in either direction. This fixed threshold reports meaningful coverage and is used by the declared offline selection policy; it is not an online gate or a searched parameter. All near-zero and zero-time-overlap parents remain.

| Dataset / setting | any improved / worse / tied | ≥0.1pp improved | ≤−0.1pp worse | positive / negative contribution pp | net pp | <−5pp |
|---|---:|---:|---:|---:|---:|---:|
| hc2 / original baseline | 69 / 54 / 5 | 34/128 | 20/128 | +0.420382 / -0.120984 | +0.299398 | 0 |
| hc2 / recommended | 92 / 33 / 3 | 60/128 | 19/128 | +0.580282 / -0.154420 | +0.425863 | 0 |
| hc2 / unconstrained mean winner | 85 / 40 / 3 | 72/128 | 23/128 | +1.203130 / -0.474598 | +0.728531 | 5 |
| vidstg / original baseline | 60 / 39 / 29 | 27/128 | 21/128 | +0.261288 / -0.196210 | +0.065079 | 1 |
| vidstg / recommended | 65 / 34 / 29 | 30/128 | 22/128 | +0.279895 / -0.179038 | +0.100857 | 1 |
| vidstg / unconstrained mean winner | 68 / 30 / 30 | 16/128 | 4/128 | +0.176894 / -0.031186 | +0.145708 | 0 |
