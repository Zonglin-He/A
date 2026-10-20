# Single ReferDINO initialization and OPD: paired bidirectional evaluation

The independent GroundingDINO stage is removed in the new executable path. One frozen native ReferDINO selected track at the original eight positions initializes E_Ref and supplies the same eight pseudo boxes for actual current-student OPD. Strong and MBR remain unchanged. All256 parents were resealed before GT evaluation. These are exposed development panels, not independent tests.

## Frozen implementation and parameters

TA-STVG Source checkpoints are unchanged in both transfer directions. HC40 and Vid30 updates; Adam0.01, tau0.2, scales(0.05,0.25),32 candidates per anchor, seed41, output alpha1. Only8×256 query-local content states are optimized; all network weights are frozen. Vid30 was selected from the prior old-initialization nine-grid. The later lr/tau search was paused before new parameter fits; no such optimum is claimed.

E_Ref uses the inherited truncated robust global four-dimensional correction, now with eight singleton ReferDINO references. It preserves the source-domain scale/truncation; no ReferDINO-specific or target-GT recalibration. Within-frame relative confidence costs are zero for singleton references. Geometric rejection affects initialization support but does not select different OPD evidence.

## Five matched rows

| Direction / method | Full vIoU % | tIoU % | Fixed actual-node sIoU % | Unobserved actual-node sIoU % |
|---|---:|---:|---:|---:|
| VidSTG_to_HC2 / Source | 25.463541 | 51.439514 | 47.979496 | 47.754147 |
| VidSTG_to_HC2 / Spatial_only | 26.661696 | 51.439514 | 49.450710 | 49.059187 |
| VidSTG_to_HC2 / Temporal_only | 26.969901 | 55.920522 | 47.979496 | 47.754147 |
| VidSTG_to_HC2 / Full | 28.253386 | 55.920522 | 49.450710 | 49.059187 |
| VidSTG_to_HC2 / Without_spatial_adaptation | 27.953988 | 55.920522 | 48.756476 | 48.525875 |
| HC2_to_VidSTG / Source | 14.582710 | 37.130256 | 36.381530 | 36.343478 |
| HC2_to_VidSTG / Spatial_only | 17.533593 | 37.130256 | 41.969520 | 41.618969 |
| HC2_to_VidSTG / Temporal_only | 16.481081 | 40.790336 | 36.381530 | 36.343478 |
| HC2_to_VidSTG / Full | 19.729106 | 40.790336 | 41.969520 | 41.618969 |
| HC2_to_VidSTG / Without_spatial_adaptation | 19.664027 | 40.790336 | 42.015608 | 41.769393 |

Five rows have the original common physical frame support and MBR. Full and Without_spatial_adaptation share Strong exactly; their difference isolates adaptation after the new E_Ref. Source and Temporal_only reproduce the prior scores exactly. All failed, zero-overlap, rejected-initialization and harmful parents remain.

## Paired differences, parent bootstrap95% intervals

| Direction / comparison | vIoU delta pp [CI95] | Unobserved actual sIoU delta pp [CI95] | Positive / negative / tied | Below −5pp |
|---|---:|---:|---:|---:|
| VidSTG_to_HC2 / Spatial_only_minus_Source | +1.198155 [-0.395612, +2.748372] | +1.305040 [-1.619617, +4.132154] | 82 / 40 / 6 | 19 |
| VidSTG_to_HC2 / Temporal_only_minus_Source | +1.506359 [+0.518667, +2.522488] | +0.000000 [+0.000000, +0.000000] | 74 / 39 / 15 | 11 |
| VidSTG_to_HC2 / Full_minus_Source | +2.789845 [+0.838341, +4.746763] | +1.305040 [-1.619617, +4.132154] | 79 / 46 / 3 | 21 |
| VidSTG_to_HC2 / Full_minus_Without_spatial_adaptation | +0.299398 [+0.058950, +0.593835] | +0.533312 [-0.035535, +1.207733] | 69 / 54 / 5 | 0 |
| VidSTG_to_HC2 / Full_minus_Spatial_only | +1.591690 [+0.556961, +2.668313] | +0.000000 [+0.000000, +0.000000] | 70 / 42 / 16 | 12 |
| VidSTG_to_HC2 / New_ERef_minus_old_E | -0.746405 [-2.692808, +1.188666] | -1.718042 [-4.932809, +1.564684] | 53 / 72 / 3 | 27 |
| VidSTG_to_HC2 / New_Full_minus_old_Full40 | -1.234752 [-3.224512, +0.761834] | -2.741040 [-6.110727, +0.611626] | 53 / 72 / 3 | 30 |
| VidSTG_to_HC2 / New_Full_minus_old_Full_matched_steps | -1.234752 [-3.224512, +0.761834] | -2.741040 [-6.110727, +0.611626] | 53 / 72 / 3 | 30 |
| HC2_to_VidSTG / Spatial_only_minus_Source | +2.950883 [+0.743720, +5.200287] | +5.275491 [+1.113210, +9.514550] | 74 / 29 / 25 | 16 |
| HC2_to_VidSTG / Temporal_only_minus_Source | +1.898371 [+0.886259, +3.099035] | +0.000000 [+0.000000, +0.000000] | 50 / 36 / 42 | 2 |
| HC2_to_VidSTG / Full_minus_Source | +5.146396 [+2.556141, +7.765879] | +5.275491 [+1.113210, +9.514550] | 77 / 30 / 21 | 17 |
| HC2_to_VidSTG / Full_minus_Without_spatial_adaptation | +0.065079 [-0.158468, +0.325617] | -0.150424 [-0.624457, +0.293745] | 60 / 39 / 29 | 1 |
| HC2_to_VidSTG / Full_minus_Spatial_only | +2.195513 [+0.941359, +3.685622] | +0.000000 [+0.000000, +0.000000] | 51 / 33 / 44 | 4 |
| HC2_to_VidSTG / New_ERef_minus_old_E | -0.518263 [-3.020921, +1.908273] | -0.796564 [-5.608323, +4.058528] | 57 / 50 / 21 | 18 |
| HC2_to_VidSTG / New_Full_minus_old_Full40 | -0.175274 [-2.639169, +2.213293] | -0.319774 [-4.953272, +4.361239] | 52 / 55 / 21 | 17 |
| HC2_to_VidSTG / New_Full_minus_old_Full_matched_steps | -0.457404 [-2.998434, +2.029637] | -0.791660 [-5.581279, +4.040171] | 52 / 56 / 20 | 18 |

Historical comparisons use the saved two-expert predictions on the exact same parents. New_Full_minus_old_Full40 changes initialization and, in Vid, the stopping budget. New_Full_minus_old_Full_matched_steps instead compares Vid30 to old Vid30; HC remains40 in both. These are historical matched controls, not new independently selected confirmations.

## Interpretation and severe cases

VidSTG_to_HC2: Full−E_Ref+Strong +0.299398pp; positive paired interval on this exposed panel. Positive contributions +0.420382pp; negative contributions -0.120984pp. Worst signed cases: RC036 -3.4926pp, RC097 -2.4445pp, RC122 -1.6264pp, RC125 -1.0846pp, RC063 -0.9376pp, RC080 -0.8530pp, RC109 -0.8170pp, RC070 -0.5094pp.
Exact mean decomposition versus the old matched-step method: initialization -0.746405pp + change in adaptation contribution -0.488347pp = complete-method change -1.234752pp. This is an accounting identity, not a causal attribution to one hyperparameter.
HC2_to_VidSTG: Full−E_Ref+Strong +0.065079pp; stable positive incremental adaptation is not established. Positive contributions +0.261288pp; negative contributions -0.196210pp. Worst signed cases: BV030 -7.1584pp, BV022 -4.5344pp, BV064 -2.8003pp, BV083 -1.6475pp, BV023 -1.4425pp, BV114 -1.3929pp, BV108 -1.0681pp, BV074 -0.7640pp.
Exact mean decomposition versus the old matched-step method: initialization -0.518263pp + change in adaptation contribution +0.060859pp = complete-method change -0.457404pp. This is an accounting identity, not a causal attribution to one hyperparameter.
BV041: new E_Ref+Strong 7.280470% → Full 7.248717%; historical old E+Strong 49.715739%, old Full30 47.835962%, old Full40 8.074698%.

No Fixed/Live contrast is added in this replacement round; gains over E_Ref measure this adaptation component, not an independent proof of on-policy superiority. Changing expert initialization can itself improve or harm outcomes, and a positive Full−Source cannot by itself establish OPD value. No new parameter search is launched from these results.

## Costs and evidence accounting

| Direction | Complete first-query wall s | Source224 s | Source448 s | ReferDINO process incl. load s | OPD fit s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | 18.597226 | 2.322676 | 2.658849 | 5.054676 | 1.133241 | 7.734084 |
| vidstg | 29.436541 | 4.147579 | 6.801538 | 6.260788 | 1.438108 | 20.194110 |

These are prelocked first-case full-input profiles, not panel-average deployment latency or a controlled speedup comparison against historical runs. Both deny detector checkpoint access and historical prediction-cache access during the new full pipeline. Their fresh Source, ReferDINO native boxes, E_Ref, Native/Strong posterior and intervals reproduce the prepared evidence exactly; profile predictions count in the256 primary fits. The runtime loads the complete ReferDINO state without its redundant independent detector-checkpoint preload; all learned parameters are covered. ReferDINO retains its native internal architecture.

Actual new fitting:256 fits /8960 updates. Worker wall HC 236.639315s and Vid 478.631169s, including recording and cache handling. Initial Source-feature reuse197;59 missing Vid Source224 caches were regenerated. Two full-chain profiles add two Source448 passes and two ReferDINO8 queries; all other teacher evidence is native-output cache reuse. Independent detector forwards:0. Prior paused-search287.901874s/40 compatibility updates are separate preparation cost; no new parameter trial was completed.

## Verification and scope

Independent enumeration verifies256 E_Ref initializations and evidence identity. All8960 actual updates have independent sampling, geometry cost, mixture probability gradient and state-step checks; zero state restores E_Ref. Official evaluator versus dense implementation checks all1280 row predictions; saved-score aggregation, parent bootstrap intervals and signed tails are independently recomputed. Raw teacher caches, video, annotations, parent IDs and original captions are excluded from public export. See RESULTS_PUBLIC.json, AUDIT_PUBLIC.json, INDEPENDENT_SPATIAL_AUDIT_PUBLIC.json, ROOT_AUDIT_PUBLIC.json and COST_PUBLIC.json.

All original predictions, negative reports and production/paper registries remain unchanged. The new package/configuration and root README define the requested one-expert research design; configuration selection does not claim independent effectiveness. Tuning remains stopped pending the result review.
