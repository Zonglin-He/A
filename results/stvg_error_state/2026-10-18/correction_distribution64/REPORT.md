# Frozen output-space correction distribution64

Latest user instruction pauses the former LN/query-residual update. Its two completed queries/four fits/160 optimizer steps are preserved and unscored. This experiment uses only existing frozen Source/E caches and the identical preselected HC32/Vid32 roster. No model, expert or optimizer call. The second backbone was explicitly deferred by the human.

## Full-tube result

All arms use the exact same final-distribution MBR time. Source is an existing same-clock reference, not an extra new variant.

| Target | Frozen Source vIoU % | E vIoU % | Correction mean % | Trajectory MBR % | Mean−E pp | MBR−E pp |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | 28.004527 | 27.558379 | 17.566493 | 21.894193 | -9.991885 | -5.664186 |
| vidstg | 16.616909 | 22.453299 | 16.257663 | 20.796704 | -6.195637 | -1.656596 |

| Target | Mean−E unobserved actual s pp | MBR−E unobserved actual s pp | MBR−E full v95% CI pp | Mean <−5pp vs E | MBR <−5pp vs E |
|---|---:|---:|---|---:|---:|
| hc2 | -16.067299 | -9.763072 | [-11.060399,-0.596978] | 18/32 | 15/32 |
| vidstg | -9.188807 | -1.885752 | [-4.756959,+0.140281] | 12/32 | 2/32 |

## Decision and scope

Neither new readout improves deterministic E in both domains. Stop this fixed correction-distribution readout; retain E. Do not search its temperature, mixture weights or new readout rules. This is not evidence against all no-network-update methods or all correction distributions. It is not OPD: it changes output-space inference only. No second backbone, network adaptation or cross-video cache starts automatically.

This random HC32 subset has E−Source = −0.446148pp, unlike the full128 positive mean; the entire fixed subset is retained. Vid E−Source = +5.836399pp. The overall256 result is not substituted into this64 table. Targets and prior calibration are historically exposed; no independent confirmation claim. Eight Vid queries have no admitted support and remain exact no-ops. Missing GT stays unknown. All negative/neutral outcomes and severe tails are in RESULTS.json.

## Exact construction and limitations

Every assignment chooses at most one retained DINO candidate per admitted frame, or pays the original outlier cutoff. Its conditional LS delta and joint assignment energy define a working Gibbs weight with the frozen source cutoff as scale. The all-rejected state has canonical Source delta0. Exact identical deltas sum their assignment mass. The global minimum matches deterministic E, but mass allocation changes the decision. There is no continuous posterior volume/Laplace normalization or calibrated identity probability.

The mean is in (cx,cy,logw,logh) correction coordinates, not averaged frame boxes. MBR selects one existing complete trajectory with maximum expected physical-cell-weighted IoU on the shared predicted event support and original sampled grid. This approximates full-tube overlap; it is not the exact dense benchmark risk. Both use precisely the same candidates and weights; no per-frame identity switching. This supplies no new visual evidence and cannot repair arbitrary local identity changes.

Mean maximum working mass/ESS:
- hc2: maximum mass 0.077047; ESS 99.800359.
- vidstg: maximum mass 0.374957; ESS 11.650332.

These are descriptive distribution diagnostics, not an independent causal diagnosis of semantic errors or a license to retune scale.

## Verification and cost

CPU readout wall 4.217920s; zero new Source forward, expert frame, network update or training. Previously paused fits are excluded from this CPU cost and disclosed separately. Cached readout cost is not first online end-to-end latency. All64 output barriers preceded new GT evaluation; old Source/E baseline metrics and all tIoUs match exactly. Independent assignment/mass/mean/utility/metric/bootstrap audits passed. Root actually viewed the two aggregate plots and first locked parent per domain at all four original planned slots; decoded original pixel SHA matched. This is engineering verification, not new semantic labels. Remote publication/archive completion are recorded by separate final receipts.

[All signed results, strata and CIs](RESULTS.json) · [Decision](DECISION.json) · [Actual protocol](ACTUAL_PROTOCOL.json) · [Costs](ACTUAL_COSTS.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Paired intervals](CORRECTION_DISTRIBUTION.svg) · [All parents](ALL_SIGNED_PARENTS.svg).


Selected-parent overlap with earlier source calibration: {'hc2': 3, 'vidstg': 0}. This was disclosed from prior metadata only, without replacing sources or selecting any score subset. The prior source-calibrated E and all negative reports remain unchanged.
