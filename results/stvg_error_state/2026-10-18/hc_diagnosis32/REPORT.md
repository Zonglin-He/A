# Why HC correct-reference correction and actual DINO correction differ

The cached evidence separates reference mistakes, geometric-coordinate errors and the scope of applying a constant correction. On the same common references, oracle repair of anchors carrying the old WI label has the largest HC category contribution; a separate geometry game assigns more of the repair contrast to center coordinates than sizes. Applying the unchanged erroneous delta outside the planned-anchor hull adds loss, but restricting it to the hull does not rescue full DINO and also discards useful correct-reference gains. These are overlapping controlled contrasts, not three independent causes or causal percentages.

## Frozen question and label provenance

The original32 exposed rollback queries, saved Before state, Source sensitivity, final-distribution MBR, four original planned positions and admitted DINO boxes are fixed. There is no new model, expert, training, Student update, pixel inspection or online replay. Final MBR remains the user’s fixed engineering component; its earlier T0 failure is unchanged. All diagnostic candidates were sealed before this run’s dense scoring. Only old sealed sparse GT entered oracle repair generation.

Old VISUAL_REVIEW labels are reused only after exact query/parent roster and panel-SHA binding: WI apparent wrong participant, CP partial/uncertain extent, MX merged instances, CI apparently consistent identity, UN uncertain. Six semantic-role-unknown queries have every slot conservatively overridden to UN while retaining the original codes. There are no new independent human labels. Existing PNG file bytes were read for SHA256 binding, but no PNG was decoded or newly visually inspected; “no new pixel review” does not mean zero binary file I/O. Identity was not inferred from IoU.

## Category-conditioned oracle repairs

DINO_common and GT_common share exactly43 HC and24 Vid available reference positions and the same scored support. For each query, every subset of categories present is replaced by its cached GT boxes; exact finite Shapley values allocate the GT_common−DINO_common contrast. The common-set filter itself uses GT availability and is not deployable. All16 parents/domain remain in primary category means, including absent-category zeros and no-reference no-ops. Category-present conditional means are secondary and use different denominators.

| HC old category | Common anchors / present parents | Full vIoU repair Shapley pp [95% CI] | Unobserved actual-decoder repair Shapley pp [95% CI] |
|---|---:|---:|---:|
| WI | 12 / 5 | +6.801663 [+1.836707, +12.667898] | +11.566492 [+2.382971, +23.645428] |
| CP | 2 / 2 | -0.126826 [-1.048785, +0.668307] | -0.228005 [-1.111286, +0.427270] |
| MX | 3 / 1 | +0.557585 [+0.000000, +1.672755] | +1.165781 [+0.000000, +3.497342] |
| CI | 15 / 6 | +0.223921 [-0.037223, +0.565350] | +0.276561 [-0.345165, +0.986463] |
| UN | 11 / 4 | +0.298320 [-3.070876, +3.615824] | -0.825663 [-4.426490, +2.976926] |

The HC total matched repair gap is+7.754662pp full vIoU and+11.955165pp unobserved sIoU. WI contributes+6.801663/+11.566492pp in this model-assisted-label game. The remaining categories have smaller positive and negative contributions; these are not causal fractions or independently certified identity effects. WI-box GT replacement jointly repairs identity and geometry for references carrying that tag; it does not isolate pure identity. Only five HC parents contain WI, and labels are not blind independent human annotations. Replacing only WI references, leaving every other DINO box unchanged, improves over DINO_common by+6.777871/+11.566516pp and yields+8.212862/+9.617499pp versus Base. Replacing only non-WI references yields+2.364199/−1.560320pp versus Base. These oracle interventions do not validate an automatic WI detector or filter.

Reference geometry is compatible with those old labels but does not recertify them. HC WI common-reference mean IoU is.043684 and target-size-normalized center error1.463254. CI has15 anchors across6 parents and mean IoU.772741, so apparently consistent identity still does not mean exact GT geometry. CP has only2 anchors in2 parents, mean area ratio.391549; MX has3 anchors in1 parent, area ratio1.641895. Their small or uncertain repair effects cannot establish that partial/merged extent error is unimportant. UN retains unknown-role and ambiguous cases, including negative cohort contributions.

Vid is context, not a second HC replication: category full-v contributions WI+1.437281, CP+.283173, MX+1.422806, CI+.085551, UN+4.050006pp; unobserved contributions+5.421643/+2.258653/+1.198197/+.256812/+7.537842. Only2/1/1/5/2 parents contain these categories. Tiny counts and old model-assisted uncertainty preclude a general identity-error ranking.

## Center and size: a different exact game

Four hybrids hold common positions fixed: DINO center/DINO size, GT center/DINO size, DINO center/GT size and GT center/GT size. They use the same constant LS and global scope. HC center/size Shapley contributions are+6.093863/+1.660799pp full v, and+10.464651/+1.490514pp unobserved s. They sum to the same total matched gap as the category game, so the two decompositions must not be added.

GT-center-only gives HC+5.967710pp full v and+6.945691pp unobserved s versus Base; GT-size-only gives+1.534645/−2.028447pp. This coordinate intervention favors correcting center coordinates in this panel. Center is not a pure identity variable: partial boxes can shift centers, and wrong participants change both center and size. Size is not all extent error. Size also interacts with center: after centers are replaced by GT, correcting size adds+3.221943pp full v and+3.060458pp unobserved s. Size-alone changes unobserved s by−.079430pp versus DINO_common, so its weak standalone contrast does not mean size is useless. The result does not identify identity as the unique underlying cause or imply that scale never matters.

## Where the same delta is applied

No delta is re-estimated when scope changes. Masks are global, the inclusive physical hull of the original four planned positions, or the original admitted decoder nodes. All three masks are defined without GT. Node-only applies the same global delta at admitted nodes; it is different from replacing those nodes with reference targets (the old LocalAnchor control). Benchmark interpolation near mask boundaries remains unchanged and contributes to full-v metrics.

| HC reference / same delta | Global Δv / unobserved Δs pp | Planned-hull Δv / unobserved Δs pp | Admitted-node Δv / unobserved Δs pp |
|---|---:|---:|---:|
| DINO_full | +0.689673 / -2.675767 | +0.165268 / -0.683930 | +0.107836 / +0.000000 |
| DINO_common | +1.434991 / -1.949017 | +0.849004 / +0.485930 | +0.210696 / +0.000000 |
| GT_common | +9.189653 / +10.006148 | +6.149697 / +3.885960 | +2.220433 / +0.000000 |

For full DINO the HC unobserved loss decomposes additively as−.683930pp inside-hull contribution plus−1.991837pp outside-hull contribution =−2.675767pp. Each part divides by that query’s original total known unobserved count before parent averaging. Conditional inside-only mean−5.387585 and outside-only mean−3.436294 use different support denominators and must not be added. Restricting application to the hull removes outside loss but leaves negative full-DINO unobserved mean; the failure is not only extrapolation.

DINO_common hull-only has+.485930pp unobserved mean, but it uses the label-availability-selected reference set and is not an unlabeled improvement. For correct GT_common the global unobserved gain+10.006148 splits into+3.885960 inside contribution and+6.120188 outside contribution. A hard hull loses the latter benefit. The diagnostic does not choose or deploy a mask. Admitted-node-only has essentially zero actual-unobserved effect by construction, while benchmark interpolation can still change.

## Limits of correct references, uncertainty and cost

Correct reference boxes do not make a constant LS fit exact at every anchor, nor optimize IoU directly. Residual losses can reflect limited constant expressiveness, averaging and chart-space/IoU mismatch; this diagnostic does not separately identify those remaining mechanisms. They must not all be called propagation failure. In saved HC examples, H003 unobserved DINO_common−34.692238 becomes GT_common−6.016689; H010−26.552276 becomes−21.826281; H005−70.001307 becomes+11.360462. These are illustrative cached-score cases, not new labels or selected evaluation parents. All original32 remain, with negative/neutral/unknown outcomes and saved-score leave-one ranges.

Source sensitivity, full and within-MBR endpoints, category-present counts, conditional reference quality, all parent signs, inside/outside supports and bootstrap CIs remain in RESULTS/DIAGNOSTICS. Reference-quality means weight anchors; primary repair effects weight parents. The10,000 fixed-seed parent intervals describe saved inherited trajectories and are not stream replay, multiplicity-adjusted confirmation or independent human-label validation.

Independent auditing reconstructs962 candidate outputs (max error6.66134e-16), evaluates500 games by full permutations instead of the runner’s subset-weight formula, verifies768 old endpoint identities and9947 scalar/aggregate/CI/quality/count/support checks. Category coordinate-error vectors also sum to the DINO-minus-GT delta independently. The largest inside/outside additive error is2.84217e-14. All original runtime/cache/registry pins remain exact.

Cached candidate generation took0.125028s and dense scoring8.931744s; this is diagnostic CPU cost, not fresh online latency. New GPU/model/expert/training/Student-update and new visual-label counts are0. The old visual panel was only byte-hashed for provenance.

The permitted conclusion is conditional: apparent wrong-participant references are a substantial oracle repair target in the old HC label game; center-coordinate repair is substantial in a separate geometric game; applying a wrong delta globally extends loss but hard restriction also removes valid gains. No causal percentage, new filter, chosen hybrid, dataset-specific recipe, production winner, OPD or automatic follow-up follows. Existing deployments and all earlier positive/negative results remain unchanged.

- [All anonymous values, exact games, Shapley, scope effects, quality and CIs](RESULTS.json)
- [Secondary contrasts, conditional counts and saved-score sensitivity](DIAGNOSTICS.json)
- [Independent audit](ROOT_INDEPENDENT_AUDIT.json), [verification](VERIFICATION.json), [actual protocol/checkpoint hashes](ACTUAL_PROTOCOL.json)
- [Aggregate figure](HC_DIAGNOSIS.svg), [locked protocol](../../../../protocols/stvg_error_state_hc_diagnosis32_v1.md)

Raw coordinates, sparse/dense GT, original query/parent text, media, q, weights and raw tensors are excluded from the public export. Old categorical labels and anonymous scalar quality measures retain their model-assisted uncertainty.
