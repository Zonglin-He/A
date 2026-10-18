# New-parent retest of frozen local teacher-endpoint update

HC16 + Vid16, one query per parent. Membership locked before cache inventory and inference. Excluded original tuning parents, old rollback/error-state32, query-only64, external20-64, soft-correspondence32, spatial-readout32, controls256 and E-expanded256. Selection used fixed salted hashes only, not scores, GT or reference quality. Historical official full-result exposure remains: this is not a globally unseen test.

Same Source checkpoints, E estimator (beta1/16/source calibration), original four anchor nodes, source exploration scales, aligned content/exclusion mask, threshold0.35, whole-state legality and final-MBR. All four complete outputs sealed before GT scoring. No training, network-weight updates, dynamic OPD or parameter search.

| Output | HC complete vIoU % | Vid complete vIoU % |
|---|---:|---:|
| E_MBR | 35.305477 | 15.755931 |
| Center_full_local | 35.484322 | 15.819496 |
| Samples24_endpoint | 35.400866 | 15.807407 |
| Center_p_local | 35.385810 | 15.805803 |

## hc2: paired changes, pp

| Contrast | Complete vIoU [95% CI] | Observed actual sIoU | Common-admitted actual sIoU |
|---|---:|---:|---:|
| Center_full_local_minus_E_MBR | +0.178846 [-0.015042, +0.431654], n=16 | +1.062015 [-0.426557, +2.713131], n=15 | +4.183538 [+0.962263, +7.815218], n=12 |
| Samples24_endpoint_minus_E_MBR | +0.095389 [+0.011352, +0.198365], n=16 | +0.861290 [+0.179695, +1.663183], n=15 | +1.925415 [+0.614256, +3.420171], n=12 |
| Center_p_local_minus_E_MBR | +0.080333 [+0.010583, +0.176739], n=16 | +0.686939 [+0.136543, +1.362088], n=15 | +1.891105 [+0.563788, +3.413381], n=12 |
| Samples24_endpoint_minus_Center_full_local | -0.083456 [-0.269968, +0.065613], n=16 | -0.200725 [-1.415511, +0.977883], n=15 | -2.258122 [-4.644967, -0.181792], n=12 |
| Samples24_endpoint_minus_Center_p_local | +0.015056 [-0.009620, +0.052012], n=16 | +0.174351 [-0.041170, +0.515968], n=15 | +0.034310 [-0.080566, +0.143031], n=12 |
| Center_p_local_minus_Center_full_local | -0.098512 [-0.275253, +0.040218], n=16 | -0.375076 [-1.521650, +0.721385], n=15 | -2.292432 [-4.661616, -0.218568], n=12 |

Known observed-node support: 15/16 queries. Common-admitted nodes with GT: 30 across 12/16 queries. Missing GT stays unknown, all16 remain in complete means.

| Coverage | Count |
|---|---:|
| planned_nodes | 64 |
| center_accepted_nodes | 43 |
| sample_any_accepted_nodes | 46 |
| sample_accepted_instances | 318 |
| common_admitted_nodes | 41 |
| common_known_GT_nodes | 30 |
| sample_legal_states | 282 |
| no_support_query | 1 |

| Output minus E | Observed v contribution pp | Unobserved actual v contribution pp | Benchmark interpolated v contribution pp | Positive / negative / neutral | Complete loss below -5pp |
|---|---:|---:|---:|---:|---:|
| Center_full_local | +0.015575 | +0.000000 | +0.163271 | 8 / 3 / 5 | 0 /16 |
| Samples24_endpoint | +0.011973 | +0.000000 | +0.083416 | 9 / 2 / 5 | 0 /16 |
| Center_p_local | +0.009273 | +0.000000 | +0.071060 | 9 / 2 / 5 | 0 /16 |

| Output | Mean observed center displacement | Mean observed log-size displacement |
|---|---:|---:|
| Center_full_local | 0.012067707 | 0.048480670 |
| Samples24_endpoint | 0.004390483 | 0.014401812 |
| Center_p_local | 0.003409792 | 0.013359890 |

## vidstg: paired changes, pp

| Contrast | Complete vIoU [95% CI] | Observed actual sIoU | Common-admitted actual sIoU |
|---|---:|---:|---:|
| Center_full_local_minus_E_MBR | +0.063564 [+0.000000, +0.139861], n=16 | +0.542814 [-1.007939, +2.018927], n=12 | +2.044130 [-1.693590, +5.383347], n=6 |
| Samples24_endpoint_minus_E_MBR | +0.051475 [-0.000809, +0.115246], n=16 | +0.461202 [-1.062067, +1.866348], n=12 | +1.832373 [-1.739245, +5.041065], n=6 |
| Center_p_local_minus_E_MBR | +0.049871 [-0.001027, +0.112790], n=16 | +0.445402 [-1.079227, +1.863604], n=12 | +1.797240 [-1.752220, +4.989147], n=6 |
| Samples24_endpoint_minus_Center_full_local | -0.012089 [-0.027305, -0.000147], n=16 | -0.081611 [-0.196406, +0.004896], n=12 | -0.211757 [-0.378620, -0.050196], n=6 |
| Samples24_endpoint_minus_Center_p_local | +0.001604 [-0.000064, +0.004240], n=16 | +0.015800 [-0.003650, +0.040078], n=12 | +0.035133 [-0.011877, +0.084867], n=6 |
| Center_p_local_minus_Center_full_local | -0.013693 [-0.031357, -0.000138], n=16 | -0.097412 [-0.220771, -0.004835], n=12 | -0.246890 [-0.436377, -0.072645], n=6 |

Known observed-node support: 12/16 queries. Common-admitted nodes with GT: 9 across 6/16 queries. Missing GT stays unknown, all16 remain in complete means.

| Coverage | Count |
|---|---:|
| planned_nodes | 64 |
| center_accepted_nodes | 22 |
| sample_any_accepted_nodes | 29 |
| sample_accepted_instances | 421 |
| common_admitted_nodes | 22 |
| common_known_GT_nodes | 9 |
| sample_legal_states | 221 |
| no_support_query | 5 |

| Output minus E | Observed v contribution pp | Unobserved actual v contribution pp | Benchmark interpolated v contribution pp | Positive / negative / neutral | Complete loss below -5pp |
|---|---:|---:|---:|---:|---:|
| Center_full_local | +0.013791 | +0.000000 | +0.049773 | 4 / 1 / 11 | 0 /16 |
| Samples24_endpoint | +0.011711 | +0.000000 | +0.039764 | 5 / 1 / 10 | 0 /16 |
| Center_p_local | +0.011245 | +0.000000 | +0.038627 | 4 / 1 / 11 | 0 /16 |

| Output | Mean observed center displacement | Mean observed log-size displacement |
|---|---:|---:|
| Center_full_local | 0.005014014 | 0.026410641 |
| Samples24_endpoint | 0.004202413 | 0.021349146 |
| Center_p_local | 0.003875002 | 0.019785141 |

## Cost and interfaces

Center_full uses a single center reference per active original anchor, no sampled admissions. Samples24 queries legal states from the initial frozen E-centered distribution once. Center_p reuses the same center output and sample admission ratio p=n/24, with no extra forward. It therefore costs sampling information and is not a low-cost center baseline. Rejected center references remain E, including in Center_p.

Common support is defined before GT by accepted center AND positive sample count. Common-node sIoU compares both outputs on that identical actual-node set. Full-tube comparisons can include differences in admission support. Equal p is not equal actual displacement norm, nor a calibrated correctness probability.

| Actual new work | HC | Vid |
|---|---:|---:|
| encoding_frames | 60 | 44 |
| encoding_native_seconds | 5.668966043042019 | 4.130919466260821 |
| center_logical_queries | 56 | 40 |
| sample_logical_queries | 1128 | 884 |
| unique_sample_queries | 1128 | 884 |
| center_decoder_seconds | 0.4778429251164198 | 0.3401876038406044 |
| sample_decoder_seconds | 0.5128962900489569 | 0.37716238806024194 |

Total new GPU-allocation wall time: 220.206149s, including frozen Source restoration and teacher worker. Teacher-worker allocation: 72.868338s. New frozen Source captures32; original cached DINO calls0. The newly counted encoding_frames are real new native DINO image/text forwards shared by center and sampled decoders. Qualification/IO/model-load overhead is included in worker allocation, not only decoder-time subtotals.

These costs reuse historical original expert inputs and E calibration. Decoder subtotals are measured after shared encoding; they are not first-use end-to-end latency. All model use was serial under the original shared GPU lease and resource guards.

## All signed complete-vIoU changes against E, pp

| Alias | Domain | Center_full | Samples24 | Center_p | Samples24 minus Center_p |
|---|---|---:|---:|---:|---:|
| NH001 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH002 | hc2 | +0.090425 | +0.021629 | +0.029593 | -0.007964 |
| NH003 | hc2 | +0.383746 | +0.164140 | +0.137768 | +0.026372 |
| NH004 | hc2 | +1.000429 | +0.061579 | +0.078117 | -0.016538 |
| NH005 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH006 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH007 | hc2 | -0.181400 | +0.035740 | +0.032065 | +0.003674 |
| NH008 | hc2 | +1.538001 | +0.653289 | +0.679492 | -0.026203 |
| NH009 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH010 | hc2 | -0.172690 | -0.000547 | -0.016436 | +0.015889 |
| NH011 | hc2 | +0.105836 | +0.436306 | +0.189651 | +0.246654 |
| NH012 | hc2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NH013 | hc2 | +0.296953 | +0.122182 | +0.142593 | -0.020410 |
| NH014 | hc2 | -0.373644 | -0.193600 | -0.140479 | -0.053121 |
| NH015 | hc2 | +0.119991 | +0.046123 | +0.051743 | -0.005619 |
| NH016 | hc2 | +0.053882 | +0.179389 | +0.101225 | +0.078164 |
| NV001 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV002 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV003 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV004 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV005 | vidstg | +0.427866 | +0.331182 | +0.325122 | +0.006060 |
| NV006 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV007 | vidstg | +0.060417 | +0.057590 | +0.058206 | -0.000616 |
| NV008 | vidstg | +0.238363 | +0.166282 | +0.147627 | +0.018655 |
| NV009 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV010 | vidstg | +0.000000 | +0.001974 | +0.000000 | +0.001974 |
| NV011 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV012 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV013 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV014 | vidstg | -0.074634 | -0.074478 | -0.074634 | +0.000156 |
| NV015 | vidstg | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| NV016 | vidstg | +0.365015 | +0.341056 | +0.341622 | -0.000566 |

## Decision boundary

This retest can distinguish center endpoints from sampled endpoints under the frozen one-round rule; it is not dynamic or on-policy adaptation. Candidate states and their teacher feedback were acquired once. Source/E and all nonobserved actual decoder nodes remain unchanged. Changes on benchmark-interpolated frames arise from the official interpolation rule, not extra visual evidence. All tIoUs match exactly.

Paired bootstrap10000, seed20261018, parent/query unit. Candidate pairs are not independent observations. Unknown/no-support/failed cases are not removed by effect. Both positive and negative findings must be retained. No dynamic OPD, new formula variant, threshold search, extra cohort or production promotion follows automatically.
