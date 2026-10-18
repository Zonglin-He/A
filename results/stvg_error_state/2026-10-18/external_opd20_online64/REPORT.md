# External20 fixed-stream OPD transfer64

Only external W4x5 updates. Backbone frozen, no new forward/expert call. Original HC32/Vid32 roster is split into16 expert and16 no-expert arrivals/domain by a score-independent locked rule, then interleaved. Main results below use NO-CURRENT-EXPERT arrivals only. Every expert arrival outputs identical E in all five arms.

## Primary no-expert results

| Arm | HC vIoU % | HC sIoU % | Vid vIoU % | Vid sIoU % |
|---|---:|---:|---:|---:|
| A_no_learning | 29.525032 | 56.259102 | 15.030949 | 33.892429 |
| B_Direct | 30.082397 | 57.552130 | 15.579125 | 34.754435 |
| C_sparse_OPD | 29.523594 | 57.084458 | 15.783216 | 35.055595 |
| D_dense_OPD | 29.451716 | 57.004470 | 15.708502 | 34.938106 |
| F_fixed_rollout | 29.310651 | 56.475431 | 15.723461 | 35.027948 |

| No-expert contrast | HC Δv pp | HC95% CI | Vid Δv pp | Vid95% CI |
|---|---:|---|---:|---|
| D−A_no_learning | -0.073316 | [-1.211488,+0.968055] | +0.677554 | [+0.202620,+1.218374] |
| D−C_sparse_OPD | -0.071877 | [-0.179408,+0.032337] | -0.074713 | [-0.168886,-0.010059] |
| D−F_fixed_rollout | +0.141065 | [-0.169836,+0.439838] | -0.014959 | [-0.148100,+0.123419] |
| D−B_Direct | -0.630680 | [-1.411778,+0.054270] | +0.129377 | [-0.096802,+0.449123] |

## Interpretation

D−A: HC−0.073316pp, Vid+0.677554pp full vIoU. There is a Vid saved-stream transfer signal; no two-domain positive full-v screen. B−A means are HC+0.557365pp and Vid+0.548177pp, so the experiment does NOT show that all external-state learning fails. B and D also have positive spatial means in both domains, but overall tube benefit differs.

D−C full v is HC−0.071877pp/Vid−0.074713pp: dense E coverage did not outperform the same teacher at four planned points. D−F is HC+0.141065pp/Vid−0.014959pp; it does not establish consistent benefit of on-policy resampling. D−B is HC−0.630680pp/Vid+0.129377pp; OPD is not established as superior to Direct. No automatic expansion or parameter sweep follows. Retain deterministic E and all limited positive cases; hold this fixed dense external-OPD variant. This is not a route-level impossibility claim.

These paired bootstrap10000 intervals are CONDITIONAL ON A SINGLE SAVED ONLINE HISTORY. They do not rerun the state sequence under resampled parents, and are not independent stream/order confidence intervals. Cohort has historical development exposure; the no-expert condition is an inference-access constraint, not untouched-data status. No target scores choose step, W or output.

## Mechanism and access verification

psi reads only frozen Source trajectory means. Existing opposite-source geometry s scales a shared unit Gaussian correction; LR.01/tau.25/5steps/M32 are common across domains. W alone carries; Adam resets per expert. C/D resample around current Wpsi each step; F freezes the first actions/rewards/weights/rollout mean. No-current-expert arrival never opens an expert input and never updates W. A same-budget baseline emits Source there. All expert outputs are exact E even if a learner fits poorly, so current expert gains cannot contaminate the primary comparisons.

Old Source archives also contain expert objects. The exporter reads only the first Source/logits raw storages and byteorder, not the pickle or expert storage; recomputes the same MBR/Source uniform positions. Thirty-two allowed expert cases independently verify raw Source/time/planned positions against their sealed E artifact; no teacher input is created for no-expert cases. The online process denies old mixed artifacts, GT/results/weights and teacher input except at the current expert arrival. See INPUT_ACCESS_AUDIT.json.

## Cost, failures and limits

CPU online run 1.386929s; actual external Adam steps 585 of maximum640, including no-reference and exact-flat reward no-update cases. Zero backbone forward/update, expert or source-training calls; cached runtime is not first online end-to-end latency.
- hc2:16 expert/16 no-expert; valid expert support16/16.
- vidstg:16 expert/16 no-expert; valid expert support14/16.

All64 outputs seal before target GT. No-reference/no-update/unknown and adverse outcomes remain in denominators. Five outputs have identical tIoU for every arrival; expert outputs match exactly. Complete no-expert/expert/all-arrival metrics, conditional CIs, negative tails, state/step displacement diagnostics and history are in RESULTS.json.

Independent audits cover chronological state links, no-expert isolation/no-update, original fixed actions vs actual resampling, reward/likelihood gradients, Direct objective and Adam moments, metrics and aggregates. An audit-only correction preserves an overly strict float64 NumPy-vs-Torch dot-product bit-equality failure; same-call recorded means remain bit-exact and independent differences are checked numerically. No input, update, prediction or metric changed.

Twenty variables and five trajectory statistics may be insufficient to transfer general corrections; this test does not identify that as the unique cause. E pseudo boxes introduce no new unobserved-frame visual evidence. Geometry feedback is not event-boundary supervision. Previous LN experiment remains user-paused; second backbone remains deferred.

[All results and diagnostics](RESULTS.json) · [Protocol](ACTUAL_PROTOCOL.json) · [Access audit](INPUT_ACCESS_AUDIT.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Costs](ACTUAL_COSTS.json) · [Decision](DECISION.json) · [Primary CIs](NO_EXPERT_CONTRASTS.svg) · [Chronological history](NO_EXPERT_HISTORY.svg).


Root actually viewed both aggregate plots and the first no-expert parent of each domain at its four Source-planned positions. Original video SHA and sampled IDs match. No current expert/GT is overlaid or consulted; no new semantic identity labels were assigned. Publication and archive closure are recorded in separate final receipts.
