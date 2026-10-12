# Q0 frozen Query observability diagnostic

The complete 64-query/three-group candidate, reward and true parameter-mean readouts were sealed before Q0 GT access. All samples are previously exposed development parents, one saved order1 Before state each. This is deterministic frozen-state exploration, not Gaussian Query OPD or a new online stream.

The one pooled output RMS budget is 0.0103996465, from the median of all 64 old Full first-step movements including zeros. Candidate amplitudes are bounded at 1.0 and unmet budgets remain in every table. Mean outputs are real decoder outputs at weighted latent parameters; candidate box averages are only Jensen diagnostics.

| Dataset | Group | Whole-tube oracle Δv (pp) | Expert top1 Δv (pp) | Parameter mean Δv (pp) | Mean unobserved decoder ΔIoU (pp) | Mean harm >20pp | Attained / tested directions |
|---|---|---|---|---|---|---:|---:|
| hc2 | Q_random | +0.067 [+0.024, +0.128]; n=32 | +0.055 [+0.011, +0.117]; n=32 | +0.005 [+0.000, +0.013]; n=32 | +0.017 [+0.001, +0.046]; n=29 | 0 | 33/512 |
| hc2 | Q_observable | +0.236 [+0.106, +0.431]; n=32 | +0.128 [+0.005, +0.327]; n=32 | +0.002 [-0.002, +0.006]; n=32 | +0.008 [-0.003, +0.020]; n=29 | 0 | 75/512 |
| hc2 | LN_observable | +1.337 [+0.903, +1.828]; n=32 | +0.796 [+0.348, +1.307]; n=32 | +0.149 [+0.075, +0.240]; n=32 | +0.350 [+0.199, +0.524]; n=29 | 0 | 320/512 |
| vidstg | Q_random | +0.102 [+0.048, +0.164]; n=32 | +0.072 [+0.027, +0.130]; n=32 | +0.002 [-0.002, +0.007]; n=32 | +0.008 [-0.005, +0.030]; n=26 | 0 | 67/360 |
| vidstg | Q_observable | +0.289 [+0.148, +0.465]; n=32 | +0.208 [+0.098, +0.346]; n=32 | -0.004 [-0.013, +0.003]; n=32 | -0.002 [-0.022, +0.023]; n=26 | 0 | 88/360 |
| vidstg | LN_observable | +0.791 [+0.414, +1.221]; n=32 | +0.209 [-0.274, +0.658]; n=32 | +0.011 [-0.015, +0.035]; n=32 | +0.027 [-0.046, +0.088]; n=26 | 0 | 257/360 |

## Direct paired group comparisons

| Dataset | Contrast | Mean-readout Δv difference (pp) | Mean unobserved decoder difference (pp) |
|---|---|---|---|
| hc2 | Q_observable_minus_Q_random | -0.003 [-0.010, +0.002]; n=32 | -0.009 [-0.031, +0.007]; n=29 |
| hc2 | Q_observable_minus_LN_observable | -0.147 [-0.238, -0.072]; n=32 | -0.343 [-0.521, -0.187]; n=29 |
| vidstg | Q_observable_minus_Q_random | -0.006 [-0.015, +0.001]; n=32 | -0.010 [-0.025, +0.003]; n=26 |
| vidstg | Q_observable_minus_LN_observable | -0.015 [-0.044, +0.013]; n=32 | -0.029 [-0.095, +0.047]; n=26 |

Candidate oracle uses one whole tube chosen by vIoU, with central Before included; it never stitches GT-best frames. Expert top1 is selected before GT with central-first stable ties. The complete anonymous rows retain no-expert/equal-reward cases, common-k and spectral energy, Jensen gaps, output-budget attainment, all readouts, all observation/support partitions and both 5pp/20pp tails.

Candidate reachability, reward identifiability and mean retention are separate claims. A small oracle restricts this finite direction set and budget; it is not an upper bound on all Query space. A positive expert top1 with a weak mean indicates projection loss. An effective Query mean without an advantage over LN does not establish Query-specific value. Wide 32-parent intervals indicate uncertainty; no arbitrary effect-size gate launches further training.

Actual autograd observed-box Jacobians and repeat outputs were checked on GPU; CPU root independently checks saved math/scalars, rewards, state restoration and dense scores. This is not an independently reconstructed complete decoder Jacobian. This experiment does not establish long-stream benefit, object identity correctness or Gaussian-policy distillation performance. Both PDFs passed actual root pixel view. Anonymous publication, remote content and archive closing are recorded by the genuine corrected-Native Q0 closing receipts; this phase does not close S/C/N/O.
