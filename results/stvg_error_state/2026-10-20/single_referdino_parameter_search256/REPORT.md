**Final numerical configuration:** the user-requested shorter HC learning rate0.014 and temperature0.14 were subsequently verified on all128 parents. See [the rounding supplement](rounding/REPORT.md) and [final five-row results](rounding/FINAL_FIVE_ROW_RESULTS_PUBLIC.json). Tables below preserve the exact-value pre-rounding selection.

# Single ReferDINO: HC and Vid finite joint parameter search

Both target datasets use the existing exposed128-parent panels and seed41. One frozen ReferDINO supplies the same eight boxes for E_Ref initialization and OPD supervision. The original gamma1 initialization, Source, Strong, MBR, policy scales(0.05,0.25),32 candidates and8×256 temporary states are fixed. Network weights are not trained. The user allowed forward computation; the search reuses valid Source/expert inputs and actually replays the frozen decoder for every new optimizer path.

## Search and selection

Coarse: lr0.003/0.01/0.03 × tau0.2/0.5;40-step paths read at10/20/30/40 and output alpha0.25/0.5/1/1.5/2. This is six optimizer paths and120 deterministic readouts per dataset. HC original40 baseline is reused after exact first-case replay; Vid40 baseline is actually run with all prior30 paths reproduced exactly. Each dataset then has one four-corner lr/tau sqrt2-neighborhood around its full-panel coarse winner, step±5 within1..40 and alpha±0.25. Coarse centers and original HC40/Vid30 configurations are retained.

Output alpha scales the final center residual linearly and width/height residual in log ratio around E_Ref. It is a deterministic output parameter, not a new training rollout. Final recommended selection keeps the original severe-loss bound and prioritizes meaningful parent coverage among configurations within0.1pp of the best protected all128 complete mean; both mean winners are retained; no video-specific stop, alpha or method switch. GT is offline after each whole phase is sealed; it is never used in adaptation.

## Selected development configurations

| Dataset | lr | tau | steps | output alpha | initialization gamma | selected phase |
|---|---:|---:|---:|---:|---:|---|
| hc2 | 0.0141421356 | 0.141421356 | 40 | 1.000 | 0.500 | protected_neighborhood/fine |
| vidstg | 0.01 | 0.5 | 30 | 1.000 | 1.000 | main/coarse |

## Initialization-strength coupling

The separate seven-strength screen sealed all256×7 predictions before offline scoring. HC gamma0.5/0.75 improved the initialization-only whole-panel score; Vid gamma1 remained best. HC therefore actually replays40-step OPD from both changed initializations at both the original baseline optimizer and original optimizer winner, then one mean-centered four-corner lr/tau neighborhood at its adapted coarse winner. Gamma scales Source-to-consensus center/log-size before adaptation; alpha scales the adaptation residual after fitting. No additional expert evidence, Source encoding or observation position is introduced. The same7200s cumulative worker cap covers all phases. The final comparison without spatial adaptation uses the same selected E_gamma, so initialization gains are not counted as OPD gains.


## Harm-informed protected recommendation

Revision002 was adopted after the retained coarse harm analysis and before final configuration selection. On HC, gamma0.5 at lr0.03/tau0.5/40/alpha1 gains only0.097783pp over lr0.01/tau0.2, but has four adaptation losses below−5pp instead of zero, including RC036−13.2231pp. A final four-corner neighborhood around the best protected center is actually run. Recommended configurations retain each domain original-baseline severe-loss-count bound (HC0, Vid1). Following the declared coverage preference before final selection, choose most parents with at least0.1pp gain within0.1pp of the protected mean maximum, then fewer0.1pp declines and higher mean; no per-video gate or output switching occurs. This is disclosed adaptive development selection, not an originally preregistered inferential guarantee.

| Dataset | recommended vIoU % | severe losses / limit | unconstrained mean winner vIoU % | its severe losses |
|---|---:|---:|---:|---:|
| hc2 | 29.013530 | 0 / 0 | 29.316199 | 5 |
| vidstg | 19.764884 | 1 / 1 | 19.809735 | 0 |

See FAILURE_PARAMETER_LINKS.md, VISUAL_FAILURE_AUDIT_PUBLIC.json, STATE_MAPPING_AUDIT_PUBLIC.json and SELECTED_FAILURE_FOLLOWUP.md for the retained harmful cases, matched parameter changes, six visually inspected examples, nonzero state-mapping audit and actual final-case outcomes.


## Coverage-prioritized selection, revision003

| Dataset | meaningful improved / harmed parents | recommended vIoU % | protected mean maximum % | concession pp |
|---|---:|---:|---:|---:|
| hc2 | 60 / 19 | 29.013530 | 29.054387 | 0.040857 |
| vidstg | 30 / 22 | 19.764884 | 19.809735 | 0.044851 |

## Complete five-row evaluation

| Dataset / row | full vIoU % | tIoU % | fixed actual-node sIoU % | unobserved actual-node sIoU % |
|---|---:|---:|---:|---:|
| hc2 / Source | 25.463541 | 51.439514 | 47.979496 | 47.754147 |
| hc2 / Spatial_only | 27.237314 | 51.439514 | 51.158189 | 50.850256 |
| hc2 / Temporal_only | 26.969901 | 55.920522 | 47.979496 | 47.754147 |
| hc2 / Full | 29.013530 | 55.920522 | 51.158189 | 50.850256 |
| hc2 / Without_spatial_adaptation | 28.587667 | 55.920522 | 50.161139 | 49.931468 |
| vidstg / Source | 14.582710 | 37.130256 | 36.381530 | 36.343478 |
| vidstg / Spatial_only | 17.533587 | 37.130256 | 42.066274 | 41.740686 |
| vidstg / Temporal_only | 16.481081 | 40.790336 | 36.381530 | 36.343478 |
| vidstg / Full | 19.764884 | 40.790336 | 42.066274 | 41.740686 |
| vidstg / Without_spatial_adaptation | 19.664027 | 40.790336 | 42.015608 | 41.769393 |

## Paired changes and signed tails

| Dataset / comparison | full vIoU delta pp [95% interval] | positive / negative / tied | worse than −5pp |
|---|---:|---:|---:|
| hc2 / Full_minus_Source | +3.549989 [+2.105160, +5.010813] | 88 / 37 / 3 | 12 |
| hc2 / Full_minus_Without_spatial_adaptation | +0.425863 [+0.160329, +0.732884] | 92 / 33 / 3 | 0 |
| hc2 / Full_minus_Spatial_only | +1.776216 [+0.728302, +2.846882] | 75 / 39 / 14 | 13 |
| hc2 / Spatial_only_minus_Source | +1.773773 [+0.760953, +2.687909] | 94 / 28 / 6 | 11 |
| hc2 / Without_spatial_adaptation_minus_original_ERef_Strong | +0.633679 [-0.317686, +1.613297] | 63 / 62 / 3 | 9 |
| hc2 / Full_minus_original_Full | +0.760144 [-0.158678, +1.719113] | 62 / 63 / 3 | 8 |

hc2: worst retained adaptation cases RC102 -4.9244pp, RC036 -3.9181pp, RC045 -1.9834pp, RC080 -1.6984pp, RC109 -1.5667pp, RC063 -1.3029pp, RC070 -1.1517pp, RC122 -0.6747pp.
| vidstg / Full_minus_Source | +5.182174 [+2.585867, +7.799148] | 77 / 30 / 21 | 17 |
| vidstg / Full_minus_Without_spatial_adaptation | +0.100857 [-0.119699, +0.362130] | 65 / 34 / 29 | 1 |
| vidstg / Full_minus_Spatial_only | +2.231297 [+0.970180, +3.724509] | 51 / 33 / 44 | 4 |
| vidstg / Spatial_only_minus_Source | +2.950877 [+0.747229, +5.186708] | 74 / 29 / 25 | 16 |
| vidstg / Without_spatial_adaptation_minus_original_ERef_Strong | +0.000000 [+0.000000, +0.000000] | 0 / 0 / 128 | 0 |
| vidstg / Full_minus_original_Full | +0.035778 [-0.023073, +0.103687] | 53 / 46 / 29 | 0 |

vidstg: worst retained adaptation cases BV030 -7.1164pp, BV108 -2.9177pp, BV064 -2.8766pp, BV022 -1.8183pp, BV114 -1.6682pp, BV083 -1.6313pp, BV053 -0.7991pp, BV047 -0.5905pp.
BV041 E_Ref+Strong 7.280470%, original Full 7.248717%, selected Full 7.249049%. The selected initialization strength is recorded explicitly; its gain is separated from subsequent OPD. No old GroundingDINO-based result is relabeled as a one-expert result.

## Evidence scope

These are development-selected results on repeatedly exposed parents. The reported paired bootstrap intervals describe the selected outcomes and do not adjust for the coarse/fine selection process. They do not establish independent generalization or a new independent Live-over-Fixed effect. Each direction has one fixed configuration shared by every video; all positive, negative, zero-intersection and no-update outcomes remain. If an adaptive winner is below E_Ref+Strong, that limitation is retained in SELECTION_PUBLIC.json rather than hidden. Source and Strong are verified unchanged. The selected E_gamma is independently verified against its initialization-strength screen; its improvement is separated from OPD relative to that same E_gamma.

All candidates and signed parent scores are public under all_parameter_outcomes/INDEX.json, including every coarse/fine setting, controls, uncertainty, tails and parameter definitions. SELECTED_FIVE_ROW_RESULTS_PUBLIC.json contains the complete selected five-row parent outcomes. This finite search does not exhaust continuous initialization strengths, sampling scales, optimizer families or longer budgets; no conclusion of global optimality is made.

## Actual search cost

- hc2 / coarse: 640 new primary fits, 25640 updates (including 40 compatibility updates), worker wall 996.569947s; peak allocated 0.912849GiB; failure=None.
- vidstg / coarse: 768 new primary fits, 30720 updates (including 0 compatibility updates), worker wall 1247.967929s; peak allocated 0.912869GiB; failure=None.
- hc2 / fine: 512 new primary fits, 20480 updates (including 0 compatibility updates), worker wall 812.539650s; peak allocated 0.912849GiB; failure=None.
- vidstg / fine: 512 new primary fits, 20480 updates (including 0 compatibility updates), worker wall 732.770974s; peak allocated 0.912869GiB; failure=None.
- hc2 / initialization_coarse: 512 new primary fits, 20480 updates (including 0 compatibility updates), worker wall 577.677574s; peak allocated 0.912849GiB; failure=None.
- hc2 / initialization_fine: 512 new primary fits, 20480 updates (including 0 compatibility updates), worker wall 772.418681s; peak allocated 0.912849GiB; failure=None.
- hc2 / protected_neighborhood_fine: 512 new primary fits, 20480 updates (including 0 compatibility updates), worker wall 776.333663s; peak allocated 0.912849GiB; failure=None.
- hc2 / selected_complete_profile: 0 new primary fits, 40 updates (including 0 compatibility updates), worker wall 25.863053s; peak allocated 7.734084GiB; failure=None.
- vidstg / selected_complete_profile: 0 new primary fits, 30 updates (including 0 compatibility updates), worker wall 38.062019s; peak allocated 20.194110GiB; failure=None.

Search costs are the full development matrix, including saving/auditing overhead; they are not single-query deployment latency. Prior Source/ReferDINO cache creation and full-input method profiles are inherited and are not free. No new expert or Source RGB encoding was necessary for the parameter paths. All network weights remain frozen; per-query temporary states and Adam are reset for each path.

The bounded optimizer coarse/fine round and evidence-driven HC initialization/optimizer supplement are complete. No automatic additional grid/seed/domain extension, checkpoint replacement or CURRENT/MAIN/PAPER promotion follows this report.

## Coverage and case interpretation

See [all-parent coverage](PARENT_COVERAGE.md) and [parameter decision](PARAMETER_DECISION.md) for the selected mean-versus-coverage tradeoffs and retained failures. In particular, Vid coverage selection retains BV030 below E by7.116367pp; the small-lr/early-stop protected mean maximum has no losses below−5pp. The recommendation does not claim every severe case was corrected.
