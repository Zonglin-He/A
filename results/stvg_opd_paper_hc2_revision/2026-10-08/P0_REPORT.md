# Fixed-parameter OPD: locked 128-parent confirmation

Each target has 128 prelocked parent sources, one query each and two orders. These parents were excluded from the current OPD parameter search but have earlier project exposure. The configurations were selected on exposed target-development GT. This is locked confirmation, not fresh unseen evaluation.

| Target / arm | Frozen vIoU % | After vIoU % | ΔvIoU pp [95% paired parent CI] | Current pp | Inherited pp | Sources worse >5 / >20 pp |
|---|---:|---:|---|---:|---:|---:|
| hc2 / on_policy | 22.156 | 23.502 | +1.346 [-0.046, +2.526] | -0.430 | +1.776 | 11 / 2 |
| hc2 / frozen_rollout | 22.156 | 22.378 | +0.222 [-0.005, +0.453] | +0.080 | +0.142 | 2 / 0 |
| hc2 / shuffled_feedback | 22.156 | 19.665 | -2.491 [-3.527, -1.470] | -2.534 | +0.043 | 31 / 2 |
| vidstg / on_policy | 14.466 | 18.337 | +3.871 [+2.430, +5.290] | +2.037 | +1.834 | 6 / 1 |
| vidstg / frozen_rollout | 14.466 | 14.477 | +0.011 [-0.411, +0.443] | +0.127 | -0.116 | 4 / 0 |
| vidstg / shuffled_feedback | 14.466 | 9.516 | -4.949 [-6.310, -3.692] | -2.419 | -2.531 | 42 / 8 |

Efficacy, feedback alignment and rollout refresh are separate comparisons. Native temporal readout is fixed: every Frozen/Before/After tIoU is identical. Source and query macro coincide here because each parent has one query and two complete orders. Full-source and official-query results still require P1.

| Target | OPD − Shuffled pp [95% CI] | OPD − Fixed rollout pp [95% CI] |
|---|---|---|
| hc2 | +3.837 [+2.257, +5.335] | +1.124 [-0.233, +2.244] |
| vidstg | +8.820 [+6.861, +10.979] | +3.860 [+2.415, +5.334] |

## Prelocked P0 gate

The prelocked both-direction stable-efficacy gate is not satisfied. Root must complete failure attribution before a decision on expensive P1. No configuration or roster changes are authorized by these results.

## Audit and cost boundaries

All 1536 adaptive fits retain full Gaussian actions/rewards/weights, gradients, Adam raw updates and all 1792 states. The CPU scorer independently recalculates support, reward/likelihood/Adam arithmetic and exact per-arm LN chains, then official and independent dense v/t readout. Root independently verified 9936 scalar/statistical/threshold/decomposition checks, including 10000 paired parent-bootstrap results. The decoder Jacobian is not independently reimplemented; arithmetic checks are not that stronger claim.

Capture is shared across matched arms and cached original DINO evidence is reused after exact input/native/interval verification. Actual new DINO calls during P0 can therefore be zero; the method still uses the nominal Uniform4 observations. Shared capture, decoder GPU fit and CPU audit are reported separately. These reused-input times are not a cold uncached end-to-end latency claim. All anonymous scalar rows and all negative tails are retained.

P0_ROOT_CASES contains deterministic scalar case choices after all outcomes were retained: strongest current correction, low-expert-quality harm, and positive expert-reward/no-current-task-gain mismatch. Identity confusion requires actual visual root inspection; low expert overlap alone does not prove identity confusion.

P1–P6 remain unexecuted at this P0 report. EATA missing direction/media/Fisher remains user paused. Existing baseline evaluation remains separately completed; old IoU-energy Ours is not the new OPD row.
