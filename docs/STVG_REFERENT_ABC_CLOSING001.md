# Valid-area native evidence A/B/C: complete matched comparison

All three groups completed both domains without a simultaneous-domain gain gate. Only Spatial WHO visual evidence selection changed. This is16 historically exposed development parents (8/domain), not48 independent sources and not confirmation. All three groups were independently published on completion; this report combines their already scored outputs, with no additional model run or GT access.

| Student arm | HC vIoU % | VidSTG vIoU % |
|---|---:|---:|
| Frozen | 27.721111 | 4.565213 |
| Temporal | 30.566418 | 5.426607 |
| Full_E1 | 30.623179 | 5.631445 |
| A | 30.690368 | 5.794307 |
| B | 30.609128 | 5.770998 |
| C | 30.491327 | 5.654642 |

| Domain | Contrast | Mean delta pp | Positive / negative / unchanged |
|---|---|---:|---|
| hc2 | A_minus_Temporal | +0.123950 | 3 / 4 / 1 |
| hc2 | A_minus_Full_E1 | +0.067189 | 5 / 3 / 0 |
| hc2 | B_minus_Temporal | +0.042710 | 3 / 4 / 1 |
| hc2 | B_minus_Full_E1 | -0.014051 | 4 / 4 / 0 |
| hc2 | B_minus_A | -0.081240 | 2 / 3 / 3 |
| hc2 | C_minus_Temporal | -0.075091 | 3 / 5 / 0 |
| hc2 | C_minus_Full_E1 | -0.131852 | 4 / 4 / 0 |
| hc2 | C_minus_A | -0.199041 | 3 / 3 / 2 |
| hc2 | C_minus_B | -0.117801 | 2 / 2 / 4 |
| vidstg | A_minus_Temporal | +0.367700 | 3 / 0 / 5 |
| vidstg | A_minus_Full_E1 | +0.162861 | 2 / 0 / 6 |
| vidstg | B_minus_Temporal | +0.344391 | 2 / 0 / 6 |
| vidstg | B_minus_Full_E1 | +0.139553 | 1 / 1 / 6 |
| vidstg | B_minus_A | -0.023308 | 0 / 1 / 7 |
| vidstg | C_minus_Temporal | +0.228036 | 2 / 1 / 5 |
| vidstg | C_minus_Full_E1 | +0.023197 | 1 / 2 / 5 |
| vidstg | C_minus_A | -0.139664 | 1 / 2 / 5 |
| vidstg | C_minus_B | -0.116356 | 1 / 1 / 6 |

A is SemVID-style paper-equation STVG adaptation, not an official upstream performance reproduction. B combines query/caption proxies, candidate coverage and state evidence. C keeps exact B ID/context/object indices and per-frame role quotas, changing only transition slots with local candidate-conditioned feature correspondence. All160/1280 visual-token policies exclude cell centers outside actual crop/image support and protect numeric IDs. Numeric glyphs are excluded from connection edges. The original native mRoPE, all nonvisual tokens, prompt, repetition1.05 and greedy96 cap are preserved. The encoder was run on original imagery, so this does not eliminate all contextual padding effects.

| Domain | Arm | Semantic same entity / identity-known | Complete target / known | Supported / contradicted / unknown role (fixed8) |
|---|---|---|---|---|
| hc2 | Full_E1 | 6/7 | 5/7 | 5 / 0 / 3 |
| hc2 | A | 5/7 | 2/7 | 5 / 1 / 2 |
| hc2 | B | 6/7 | 3/7 | 6 / 0 / 2 |
| hc2 | C | 6/7 | 3/7 | 6 / 0 / 2 |
| vidstg | Full_E1 | 2/4 | 2/4 | 0 / 2 / 6 |
| vidstg | A | 3/4 | 2/4 | 1 / 1 / 6 |
| vidstg | B | 4/4 | 2/4 | 2 / 0 / 6 |
| vidstg | C | 4/4 | 2/4 | 2 / 0 / 6 |

The semantic panel is fixed first four predetermined sources/domain and slots0/3, actually inspected and locked before reading new choices, by one project-exposed annotator. Partial identity, complete extent and role support remain separate. Ambiguous referents remain unknown. A role-consistent choice is not evidence of an internal role-reasoning mechanism. The panel cannot establish generalization or replace the full Student endpoint.

| Domain | B/C windows | Changed transition token sets | Changed WHO choice |
|---|---:|---:|---:|
| hc2 | 32 | 32 | 7 |
| vidstg | 32 | 32 | 13 |

## Observed evidence-transfer bottlenecks

B improves the fixed-panel same-entity/role-supported counts relative to A in both domains, but does not exceed A in complete Student vIoU. In Vid, B/C select a partial target in two of the four identity-known windows and a complete target in the other two. Better entity identity is therefore not automatically better geometric extent. Six of eight Vid role windows remain unknown in every relevant B/C count; this is not broad role-understanding certification.

C changes transition selections in all64 windows, and changes WHO choices in7/32 HC and13/32 Vid windows. The fixed-panel identity/role labels remain exactly the same as B. C-B actual Student vIoU is negative in both domains, as shown above. This isolates a failure of this fixed connection selector to add Student benefit; it does not erase the positive Vid-versus-Temporal outcome or refute all temporal/instance connections.

For a concrete posthoc trace, previously exposed diagnostic source R010:

| Arm | Two inspected role outcomes | Anchors | Accepted steps | Final parameter change | Student vIoU % |
|---|---|---:|---:|---|---:|
| A | contradicted / supported | 3 | 3 | True | 13.445906 |
| B | supported / supported | 4 | 0 | False | 13.259440 |
| C | supported / supported | 2 | 0 | False | 13.259440 |

Its B selection corrects an inspected wrong-role choice, yet the original full-step acceptance rejects the update and leaves the Student at Temporal. A has an accepted update and a small additional Tube gain despite a worse inspected role choice. This example separates semantic choice, admitted geometric supervision, accepted parameter change and final task readout. It is an illustration from an exposed panel, not a new benchmark or proof that acceptance should be removed.

The measured choice is to retain A as the strongest observed full-Student comparator in this finite run, preserve B as an identity/role evidence clue, and not claim incremental value for current C connectivity. This does not silently deploy A, select a different recipe per domain, abandon Spatial or impose a simultaneous-domain positive-gain gate. Further changes require a separate controlled test; this three-arm execution itself is complete.


Full_E1 retains all original visual tokens, so its contrast is not a padding-only or equal-budget pruning ablation. Equal budget is enforced across A/B/C. A→B is a complete evidence-policy comparison, with several coordinated differences; C→B isolates transition selection under identical fixed/object tokens and budgets. It does not isolate attention-based semantic connectivity: these are projected-feature proxies, not SemVID attention CS/ER. Existing captions are Source-conditioned and can share errors with the Student. E1 is the matched cached unpruned R1 control, not historical DINO Online OPD. Temporal512 and Student1792/lr/steps/sigma/tau/rejection/episodic-reset settings are unchanged; original early stop remains active. Every nonzero update is followed by actual complete-model decoding.

Five metrics, every positive/negative/neutral/unknown source, selection/admission/no-update state and common-observation geometry are supplied in individual groups and this combined scalar export. Common support is not always actual gradient support. Missing exact GT positions remain unknown; no GT candidate or Oracle fit is used. Saved mathematics/Adam/response/acceptance/readout roots pass before postseal official and independent metric paths. No second full backward is claimed.

All actual stage/load/generation/selection costs, request token counts, exact A partial-response reuse and B/C cached-encoder reuse appear in group cost files. Token retention is not an8x end-to-end speed claim. The initial empty-frame capacity assertion and immutable002 correction are retained; no science parameter was changed. Media, GT coordinates, private Query/caption text, raw Teacher output, vectors, states, weights and personal conversations are excluded. No Bootstrap, CURRENT change, official fullset, automatic larger sweep or new monitor follows from completion. All results, including domain differences, remain available; no per-domain winner has been silently promoted.
