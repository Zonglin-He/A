# Matched reference intervention: DINO/GT oracle × Direct/original OPD

**All four arms are GT-assisted diagnostics, not deployable TTA results.** DINO arms also use exact-GT availability to restrict support. Target-GT coordinates are used only in C/D. No arm,step,reference or method is selected by these evaluation metrics.

Same exposed HC16/Vid16 development cohort,67 common admitted anchors;25 nonempty sources and7 exact no-ops. No added/interpolated GT positions. All four arms reset to identical frozen Source1792 state and fresh Adam; no LN carry. Native time is fixed across arms; saved frozen Source projection is a secondary CPU readout. DINO references are reused,0 new Teacher calls.

| Target | Subset | n | Source | A DINO Direct | B DINO OPD | C GT Direct | D GT OPD |
|---|---|---:|---:|---:|---:|---:|---:|
| hc2 | all32 | 16 | 18.831315 | 20.125846 | 21.622803 | 31.137299 | 27.540589 |
| hc2 | nonempty25 | 15 | 20.086736 | 21.467569 | 23.064323 | 33.213119 | 29.376629 |
| vidstg | all32 | 16 | 7.147781 | 11.278990 | 11.189995 | 17.291448 | 15.576799 |
| vidstg | nonempty25 | 10 | 9.381650 | 15.991585 | 15.849193 | 25.611518 | 22.868080 |

Values are vIoU percentages, primary fixed Native interval. Full tIoU/sIoU/R@.3/R@.5 and observed/unobserved-native/interpolated geometry are retained in RESULTS.json. All32 and the same25 support-bearing sources are separate; omitted supervision is not silently filled.

| Target | Subset | Contrast | Mean ΔvIoU pp | Paired t95 | Positive / negative / zero |
|---|---|---|---:|---|---|
| hc2 | all32 | C-A | 11.011453 | [5.824971, 16.197935] | 14/1/1 |
| hc2 | all32 | D-B | 5.917787 | [-1.672829, 13.508403] | 11/3/2 |
| hc2 | all32 | B-A | 1.496956 | [-2.847031, 5.840944] | 10/5/1 |
| hc2 | all32 | D-C | -3.596710 | [-9.412440, 2.219020] | 3/12/1 |
| hc2 | all32 | (D-B)-(C-A) | -5.093666 | [-11.395880, 1.208547] | 4/11/1 |
| hc2 | nonempty25 | C-A | 11.745550 | [6.426227, 17.064872] | 14/1/0 |
| hc2 | nonempty25 | D-B | 6.312306 | [-1.802920, 14.427532] | 11/3/1 |
| hc2 | nonempty25 | B-A | 1.596754 | [-3.070617, 6.264124] | 10/5/0 |
| hc2 | nonempty25 | D-C | -3.836491 | [-10.068467, 2.395486] | 3/12/0 |
| hc2 | nonempty25 | (D-B)-(C-A) | -5.433244 | [-12.167892, 1.301404] | 4/11/0 |
| vidstg | all32 | C-A | 6.012458 | [1.460426, 10.564490] | 9/0/7 |
| vidstg | all32 | D-B | 4.386804 | [0.611346, 8.162263] | 7/2/7 |
| vidstg | all32 | B-A | -0.088995 | [-0.636616, 0.458626] | 4/5/7 |
| vidstg | all32 | D-C | -1.714649 | [-3.204928, -0.224369] | 2/7/7 |
| vidstg | all32 | (D-B)-(C-A) | -1.625654 | [-2.870629, -0.380679] | 1/8/7 |
| vidstg | nonempty25 | C-A | 9.619933 | [3.100086, 16.139781] | 9/0/1 |
| vidstg | nonempty25 | D-B | 7.018887 | [1.334578, 12.703196] | 7/2/1 |
| vidstg | nonempty25 | B-A | -0.142392 | [-1.089212, 0.804428] | 4/5/1 |
| vidstg | nonempty25 | D-C | -2.743438 | [-4.994287, -0.492590] | 2/7/1 |
| vidstg | nonempty25 | (D-B)-(C-A) | -2.601046 | [-4.393573, -0.808520] | 1/8/1 |

C-A and D-B isolate the reference-coordinate intervention within each updater. B-A and D-C compare the fixed updater configurations on matched starting state/support/time/max rounds. The interaction is (D-B)-(C-A). Intervals are descriptive on already exposed development sources; they do not establish independent confirmation.

## Mechanistic findings

Better benchmark reference coordinates improve both updater means: C-A is+11.011453/+6.012458pp HC/Vid, and D-B is+5.917787/+4.386804pp. Direct C-A has14/1/1 positive/negative/zero HC sources and9/0/7 Vid. OPD D-B has11/3/2 and7/2/7; correct references are not sufficient to ensure a beneficial parameter update.

On DINO support, OPD-Direct is+1.496956pp HC and−0.088995pp Vid; both descriptive t95 intervals cross zero. With oracle references, OPD-Direct is−3.596710pp HC (t95[−9.412440,+2.219020]) and−1.714649pp Vid ([−3.204928,−0.224369]). This fixed-budget intervention does not establish a general superiority or robustness claim for OPD. Direct has higher oracle-reference point estimates in both domains; no production method is changed based on these exposed oracle results.

H002 now supplies a matched correct-reference failure: Source vIoU31.585462%,GT-Direct46.978912%,GT-OPD8.870489%. GT-OPD observed mean IoU improves57.151969→64.145826%, while unobserved native mean drops47.242920→27.407058%; on the same Native readout support it drops53.261508→25.296456%. One correctly supervised anchor switches to the interaction counterpart by step8,its GT IoU and all32 local sampled rewards become zero; the remaining anchors continue updating. Final three anchor IoUs are0/.948212/.976163. Thus this is not a claim that every observed anchor improves: average observed improvement conceals one destroyed anchor. All GT references were fixed throughout; no best-step or oracle acceptance is introduced.

A second limitation is local reward coverage. DINO-OPD H005 has40 no-information rounds and preserves Source against a disjoint bad reference; H016 also has40 such rounds under both DINO and GT references and misses a useful correction. GT-Direct H016 gains22.110535pp while GT-OPD makes no update. All sampled IoU rewards in these no-update fits are zero. The same mechanism can suppress wrong supervision and fail to use correct supervision; it is not evidence that OPD recognizes semantic reliability. Vid GT-OPD V012 also has10 no-information rounds. Full UPDATE_DIAGNOSTICS and saved-path metrics retain these cases.

The DINO row differs from the earlier all-anchor comparison because all39 event-exterior references are excluded from every arm here by GT availability. Its new positive mean cannot justify a deployable event filter; neither the selection mask nor this oracle row is available to target-unlabeled TTA. Earlier Direct versus inherited-OPD scores remain distinct and unchanged.

Decision: reference quality is a substantial bottleneck in this cohort, and the correctly supervised H002 failure plus flat-reward cases show a separate update/coverage bottleneck. No new critic, identity architecture, optimizer modification or parameter sweep is automatically launched. A deployable reference-reliability mechanism and matched evidence of update value would still be required before method promotion.

HC lr.01/40 rounds;Vid lr.03/10,Adam(.9,.999),eps1e-8,1792 parameters. Direct preserves original5L1+2GIoU sum/4; OPD preserves original Gaussian32 antithetic,detached IoU/control-variate and per-supported-frame mean,HC sigma.025/tau.05,Vid sigma.1/tau.25. Same seeds in B/D; first samples match exactly. Maximum update rounds are matched, while original flat-reward OPD skips remain and actual gradients are counted. No rollback,best-step selection,response filter,annealing or hyperparameter search. These are fixed-budget configurations,not claims about optimally tuned algorithms.

All128 logical outputs sealed before comparative evaluation. Source zero output exactly matches the retained Source; nonempty fits match complete two-decoder readouts. All arm starts and Source model hashes are checked. GT assistance is declared in every payload. The legacy OPD arithmetic checker receives its generic original fitter flag in a temporary math-only view; this does not change the true GT-assisted provenance or make an oracle arm target-unlabeled.

Independent checks: 1800 scalars/means, 229376 matched initial coordinates, 50 complete OPD Gaussian/Adam path audits, 100 Direct endpoint losses;124 original DINO RGB SHA matches. Fixed H001/H002/V014 actual output panels are privately viewed; no new independent human semantic labels. Source/Teacher/fit/reference/GT coordinates and media are not public.

No Source/Teacher architecture change, critic training, identity model, temporal adaptation, state writeback or old-queue resume. CURRENT/MAIN/PAPER stay unchanged. Benchmark GT remains subject to the annotation uncertainty already disclosed in diagnosis32. This oracle intervention is not a deployment recipe, and its results must not be used as target-GT-based tuning or acceptance gates.
