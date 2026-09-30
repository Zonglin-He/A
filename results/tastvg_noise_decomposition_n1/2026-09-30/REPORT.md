# N1: GT-audited teacher preference noise decomposition

**Mechanism oracle; not deployable TTA.** GT filters the original teacher signs on 16 already-exposed sources. The five main arms use pairwise softplus, not the frozen method's reverse KL. No method promotion or Paper48 restart.

Five shared-source orders; each order has clean and five 5% transient conditions. Values below are corruption source-macro percentage-point changes versus Frozen, mean ± sample SD across orders. Orders share the same sources.

| Arm | Future ΔsIoU pp | Future ΔvIoU pp | Whole ΔvIoU pp |
|---|---:|---:|---:|
| All (pairwise reference) | +0.023387 ± 0.012347 | +0.006063 ± 0.003893 | +1.501587 ± 0.863206 |
| Useful-only | +0.033835 ± 0.013469 | +0.008615 ± 0.004104 | +1.504476 ± 0.863418 |
| Noisy-only | +0.003345 ± 0.012331 | +0.001029 ± 0.003076 | +1.496799 ± 0.863499 |
| Useful-Positive | +0.039819 ± 0.013959 | +0.010084 ± 0.004261 | +1.505989 ± 0.862357 |
| Useful-Negative | +0.018733 ± 0.011373 | +0.004773 ± 0.003496 | +1.500462 ± 0.862937 |
| Useful-Matched | +0.020960 ± 0.014136 | +0.005162 ± 0.003477 | +1.501458 ± 0.863856 |
| Noisy-Matched | +0.003041 ± 0.012133 | +0.001041 ± 0.003027 | +1.496801 ± 0.863571 |
| Frozen recipe RKL-Final (context) | +0.094390 ± 0.071792 | +0.025505 ± 0.020209 | +1.517681 ± 0.859943 |
| Temporal Fast-only (context) | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +1.496239 ± 0.863637 |

## Signal exposure during corruption streams

Counts pool the five order repeats; they are not independent signal samples. Supports and labels are recomputed at each arm's own evolving state.

| Arm | Selected signals | Updated arrivals | Mean gradient norm on updates |
|---|---:|---:|---:|
| All (pairwise reference) | 632 | 79 | 0.6099374309554632 |
| Useful-only | 454 | 79 | 0.9797342019772655 |
| Noisy-only | 176 | 51 | 1.1878609557723874 |
| Useful-Positive | 234 | 79 | 1.091392149753166 |
| Useful-Negative | 220 | 69 | 0.6179341077965663 |
| Useful-Matched | 135 | 51 | 1.2133241648753834 |
| Noisy-Matched | 135 | 51 | 1.1845933051274358 |

## Historical Tier-0 check

Frozen S0.6 support: corruption 576 comparisons, 568 decisive, correct positive/negative 208/208 and noisy positive/negative 78/74. Noisy fraction 26.7606%; clean 25%. These are source-state statistics, not assumed to apply unchanged to adapted trajectories.

## Interpretation limits

Useful/Noisy retain Sa2VA's original signs; GT chooses membership only. All retains every teacher-decisive signal, including GT ties. No valid spatial expert or empty subset means no update. The supplementary pair chooses the same count at each scheduled arrival, using the smaller of its two current eligible pools and a fixed hash. It subsamples Noisy when required and must not be confused with the primary unfiltered Noisy arm.

Each selected loss is averaged over its subset. SGD is fixed at .005, without gradient-norm matching or loss-scale tuning. Differences can include changed gradient scale, state-dependent eligibility and empty-update frequency. Exact-count matching addresses signal counts/update opportunities but does not make the state trajectories equal.

The within-N1 All arm is the positive control for these pairwise-loss statements. Comparing its absolute effect to historical RKL does not establish RKL's mechanism. In particular, low Noisy effect cannot establish that RKL absorbs noise, and positive Noisy effect alone cannot identify self-regularization. GT-filtered oracle gains do not demonstrate a GT-free reliability rule.

Paper48 remains paused at temporal 1039/2658 and spatial 2658/2658. Its original budget files and all completed artifacts are preserved. No baselines, new models or hyperparameter sweeps were launched.
