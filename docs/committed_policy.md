# HC64 committed-source policy trial

Research only; the registered method and paused formal/CG jobs remain unchanged.
This is a new mechanism and actual adaptation trial, not an equivalent speedup.

The previous frozen-state intervention changed all237 immediate memories but only3/2814 later actions and none of2276 nodes four or more steps away. Locally helpful forced corrections did not change their subsequent actions. Teacher suggestions also harmed72/174 scored intervention nodes. The trial tests stronger persistent decision state and external consistency weighting together; it cannot isolate their separate causal contributions.

## Fixed experiment card

Reuse the exact64 exposed development parents, seed41, saved ERef/Strong, Source features, same eight ReferDINO observations and the immutable three-slot complete-box bank. No model/decoder/expert/RGB call, temporal change, new candidate or network-weight update occurs. All outputs and update paths for all64 must seal before the independent scorer reads GT. No diagnostic GO gate precedes the real fits.

For each observation source, calculate other-anchor full-box transformation/ROI agreement on56 directed pairs. Mean incoming/outgoing agreement has fixed denominator14 and excludes self evidence. Candidate support multiplies source quality but retains the old denominator. A source may write a commitment only with quality>=.1, at least two other mutual witnesses>=.1 and positive candidate support relative to E. These are fixed uncalibrated consistency rules, not semantic verification. Same-eight-frame teacher context is correlated; eight consistent wrong-instance observations remain unresolved.

Replace appearance EMA by nine source states: E plus eight observation sources. An admitted selected alternative writes its source, which persistently biases later compatible admitted alternatives by2; E does not erase it. A later externally verified contradictory observation clears the commitment. Source is provenance, not a proven instance identity. Direct uses the differentiable probability mixture of exactly these transitions and receives the same evidence, gates, reset rules and logits.

Four positive bounded query-local weights start at1: two evidence-delta coefficients2,2, incoming-link coefficient.25, guard coefficient−.1. No learnable E bias or reversed geometry penalty. Student prior E.7, alternatives share.3. Teacher feedback is exact Q4 under the current student with rho=active*(delta_support+.25*(incoming_link−link_EE)−.1G), active edges only. Commitment itself receives no reward. Teacher probabilities include the same prior and temperature.2.

OPD acquires two current hard histories per outer; FixedKD retains two initial teacher histories but refreshes feedback each outer. Both reuse targets for ten inner updates. Direct recomputes soft histories and Q4 every inner update, retaining all available history/continuation/target gradients. Three rounds × two paths × ten Adam steps, lr.1, clip1, prior KL.05. Every method performs30 actual backwards. After ten steps an actual greedy utility decrease restores raw weights and all Adam state, then proceeds to the next declared round. GT does not choose any acceptance or parameter.

Report separate initial-policy and exact no-learning additive-utility DP controls plus sealed ERefStrong, GeometryOPD, BoxFixed and old B_OPD. The explicit finite-state environment can also be planned; OPD necessity cannot be inferred from merely improving a weak learned control. Primary outcomes are paired full vIoU and fixed unobserved-frame sIoU; include signed/severe regressions, real commitment/switch/reset and replacements, teacher suggestion harm on saved final states, actual accepted rounds and cached-only cost. Single locked configuration; no sweep or post-score changes.

## Reproduce

Use the existing private cached assets and environment. `python -B scripts/committed_policy/run.py --run`, then `python -B scripts/committed_policy/score.py --run-after-seal` after the64-query seal. Input/source/receipt hashes are enforced. Outputs are independent of all previous trials. Raw caches, paths and queries stay private; public results are anonymous aggregate/signed metrics and reproducibility code.
