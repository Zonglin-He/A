# Observation-response matched update: actual technical qualification

**Decision: stop this fixed recipe before the 192-fit panel.**

Eight fixed historical queries (HC 4 / Vid 4), two fresh captures each, 16 accepted complete original-A replays, and 144 local P/F/R proposals at first/middle/last A states are complete. The fixed 64-query candidate panel and all candidate GT efficacy scoring have not run. These local probes are not complete candidate fits.

A/N source, inherited state, Omega0 observations, admitted Top1 evidence, pixels, original Native WHEN, noise and configurations remain fixed. P/F use 1792 active parameters; R uses the once-per-query entering-state curvature-trace scale.

| Dataset | F actual checks accepted / informative | FD tests pass / total | Entire FD contexts pass | AD JVP / VJP | Repeat mismatch |
|---|---:|---:|---:|---|---:|
| hc2 | 24/24 | 8/48 | 0/12 | True / True | 2 |
| vidstg | 16/18 | 17/36 | 2/9 | True / True | 0 |

## Interpretation

All matrix rows cover the complete original parameter interface and selected output grid. However, an automatic-differentiation Jacobian of the native backward graph is not automatically the derivative of the complete physical output replay. The unchanged PosDecoder sets each next reference box to `new_pred_boxes.detach()` (original query_decoder.py line 189). FD changes those forward reference values whereas the backward graph cuts their derivative. This is an identified distinction in the implementation; we do not claim this source line alone proves every individual FD discrepancy or exclude activation-boundary effects.

The complete fixed FD checks and repeat outcomes are reported, including failures. Actual F counterfactual response/drift checks can pass despite this derivative discrepancy. Such local feasibility does not establish a qualified full functional Jacobian, GT retention, error correction, or method performance. We do not change the decoder gradients, frames, precision, thresholds, or cohorts to obtain a pass.

The attachment identity g=J_O^T h is valid for the *native AD graph* used by OPD. Its J_O cannot simultaneously be assumed to be the FD-consistent full output-function Jacobian when stop-gradient reference paths differ. The promised first-order true functional guarantee therefore has not been established for this implementation.

A P repeat can differ at float32 last bits when the diagnostic graph includes U rows in one capture and only observed rows in the other. Those repeat failures are retained as technical negatives; they are not omitted or converted into an exact-repeat success. All original complete-A replays are bitwise exact.

The two earlier state/optimizer storage dispatch mistakes and the initial repeat-abort boundary were preserved in separate recovery namespaces. Exact proposal state reinsertion avoids float32 subtract/add round trips; deep-cloning saved Adam snapshots avoids shared CPU `step` tensor mutation. On-disk original fits, scientific configuration, source model and baseline processes were preserved. The final accepted prefix is exact-rechecked; execution included 8 additional original complete replays/captures before this final bounded qualification (24 actual complete A replays in total), disclosed separately from the 16 accepted ones.

## Costs and undefined paths

| Dataset | P median seconds | F median seconds | R median seconds | No-op local probes |
|---|---:|---:|---:|---:|
| hc2 | 0.222294 | 1.307633 | 0.080900 | 0 |
| vidstg | 0.299352 | 3.541048 | 0.096188 | 18 |

These are real repeat-1 local-proposal costs, excluding repeat-0 directional FD instrumentation. They include each operator's Jacobian, solver, counterfactual output checks and CPU arithmetic/recording audits. They are not deployment throughput, full adaptation latency, or a new DINO benchmark. No new DINO call or GT read occurred. Empty evidence keeps the original no-op and unknown denominators remain null.

## Independent physical derivative readback

A separate finite root pass made 8 additional fresh captures on the same fixed queries, with 21 original-A state contexts and no optimizer step or new candidate fit. It saved actual plus/minus outputs and live VJP contractions, then independently recomputed all 84 FD checks and 16859136 stored AD-Jacobian coordinates on CPU. Exactly 25/84 FD checks passed, reproducing the qualification failure values. Total actual fresh captures are 32: 16 accepted qualification captures, 8 earlier engineering rechecks, and 8 independent root captures. Total complete original-A replays remain 24; there are no candidate formal fits or GT scores.

## Scope and remaining obligations

Only this fixed numerical/technical qualification is closed. Formal P/F/R 192 fits, candidate GT evaluation, Source/order2 replication, independent new-parent validation and a newly generated online stream are unexecuted. Their conditional gate was not reached. A different complete-function derivative construction would be a material new qualification decision, not a silent fix to this result. This negative result does not reject every projection or functional-protection method.

P7-A stays frozen; CURRENT_METHOD remains the old Spatial OPD; remaining formal baselines remain cancelled with the same stopped process/optimizer/lease identity. No paper, new teacher, gate, full main run or production promotion was performed. The existing monitor remains ACTIVE because the future method/main protocol is unresolved.

Primary context: [EATA](https://proceedings.mlr.press/v162/niu22a.html) and [NTK-overlap analysis](https://proceedings.mlr.press/v130/doan21a.html) already discuss parameter protection/projected interference; this qualification claims no novel general projection theory.

All anonymous local outcomes, FD directions, no-information cases and measured costs are included in QUALIFICATION_ROWS.json, FINITE_DIFFERENCE_ROWS.json and TECHNICAL_STATISTICS.json. Private Jacobians, vectors, boxes, input caches, media, annotations, weights and conversation records are excluded from publication.
