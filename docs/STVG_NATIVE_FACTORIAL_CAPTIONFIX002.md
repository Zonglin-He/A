# True Native-clock Spatial-only and Temporal-conditioned evidence factorial

Already exposed development cohort:128 independent parents per domain,256 total. Confirmation128 remains a completed failed B-versus-A confirmation and is not used for new selection. Native A and B rebuild observations, candidates and Teacher references at the Source Native interval and execute real episodic spatial adaptation. Existing available Native-candidate Captions are reused and54 previously unrequested valid clips are generated separately (35 core and19 confirmation; plus2 Caption/2 alignment qualification calls), without new Temporal-parameter updates. For43 parents with identical Native and Adapted intervals, exact original E1 candidate pools and marked pixels are reused; other213 use the prepared Native inputs. Original Native feature work completes before handoff and is reused; corrected feature preparation adds only16 visual encodes and reuses1008 existing visual features. Initial feature-stage total wall time was not captured by the stopped parent and remains unknown; its recorded encoder/model-load counters are preserved separately. Shared generation and all new controller stages are included once in the cost table. Frozen, Temporal and Full-A are reused exactly. Full-B uses the old221 complete outputs plus35 actual Native-B full outputs only after exact same-clock E1 jobs, Source, Temporal interval and recipe checks. Historical original Full-B remains a separate row; the new comparison explicitly repairs Caption acquisition on both clocks.

| Domain | Arm | vIoU % | tIoU % | dense known-GT sIoU % | R(vIoU>=.3) % | R(vIoU>=.5) % |
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|21.651160|39.546835|53.809342|30.468750|7.812500|
|hc2|Temporal|26.293204|49.042562|53.809342|41.406250|13.281250|
|hc2|S_A_Native|22.163967|39.546835|54.983299|31.250000|9.375000|
|hc2|S_B_Native|22.131307|39.546835|54.903673|32.812500|8.593750|
|hc2|T_S_A_Native_evidence|26.906886|49.042562|54.983299|41.406250|12.500000|
|hc2|T_S_B_Native_evidence|26.854657|49.042562|54.903673|42.187500|13.281250|
|hc2|Full_A_Adapted|26.487575|49.042562|54.307313|41.406250|11.718750|
|hc2|Full_B_Adapted|26.646800|49.042562|54.671292|42.187500|11.718750|
|hc2|Full_B_original_missingCaption|26.648599|49.042562|54.660007|42.187500|11.718750|
|vidstg|Frozen|9.843861|32.374371|30.729497|10.156250|1.562500|
|vidstg|Temporal|10.328918|33.602694|30.729497|10.937500|1.562500|
|vidstg|S_A_Native|10.728285|32.374371|32.257942|10.937500|2.343750|
|vidstg|S_B_Native|10.262778|32.374371|31.360643|9.375000|3.125000|
|vidstg|T_S_A_Native_evidence|11.247888|33.602694|32.257942|12.500000|2.343750|
|vidstg|T_S_B_Native_evidence|10.761380|33.602694|31.360643|10.937500|3.125000|
|vidstg|Full_A_Adapted|10.661083|33.602694|31.154430|12.500000|1.562500|
|vidstg|Full_B_Adapted|11.059517|33.602694|31.977050|13.281250|3.125000|
|vidstg|Full_B_original_missingCaption|11.137127|33.602694|32.044467|13.281250|2.343750|

| Domain | Policy | Contrast | vIoU delta pp | Paired source 95% t interval pp | + / - / neutral / unknown |
|---|---|---|---:|---|---|
|hc2|A|Temporal_independent|+4.642044|[2.9907004138258935, 6.2933879382533835]|79/23/26/0|
|hc2|A|Spatial_Native_independent|+0.512807|[0.23331225303674435, 0.7923021635541774]|68/35/25/0|
|hc2|A|Spatial_given_T|+0.194371|[-0.06035025017028625, 0.4490929662076132]|64/41/23/0|
|hc2|A|factorial_interaction|-0.318436|[-0.6344991283944351, -0.002372572159159647]|50/51/27/0|
|hc2|A|Adapted_minus_Native_evidence_fixed_T|-0.419311|[-0.7393968162281473, -0.09922433550610277]|44/58/26/0|
|hc2|B|Temporal_independent|+4.642044|[2.9907004138258935, 6.2933879382533835]|79/23/26/0|
|hc2|B|Spatial_Native_independent|+0.480147|[0.19398853922952847, 0.7663056049442276]|66/38/24/0|
|hc2|B|Spatial_given_T|+0.353596|[0.034919095936104094, 0.6722726360145159]|71/36/21/0|
|hc2|B|factorial_interaction|-0.126551|[-0.4120675575785745, 0.15896514535543838]|61/41/26/0|
|hc2|B|Adapted_minus_Native_evidence_fixed_T|-0.207857|[-0.47491859123873303, 0.05920393325345161]|52/50/26/0|
|vidstg|A|Temporal_independent|+0.485057|[-0.014067239077919291, 0.984181490611901]|37/25/66/0|
|vidstg|A|Spatial_Native_independent|+0.884424|[0.22445357807021105, 1.5443944879991163]|55/25/48/0|
|vidstg|A|Spatial_given_T|+0.332165|[-0.2933491920105996, 0.957680103457688]|41/28/59/0|
|vidstg|A|factorial_interaction|-0.552259|[-0.9984851458683133, -0.10603200875392543]|26/40/62/0|
|vidstg|A|Adapted_minus_Native_evidence_fixed_T|-0.586805|[-1.0153134165623168, -0.1582964411980694]|24/42/62/0|
|vidstg|B|Temporal_independent|+0.485057|[-0.014067239077919291, 0.984181490611901]|37/25/66/0|
|vidstg|B|Spatial_Native_independent|+0.418917|[-0.0597510518792474, 0.8975850390333198]|43/29/56/0|
|vidstg|B|Spatial_given_T|+0.730600|[0.13877985204072674, 1.3224192521805467]|45/24/59/0|
|vidstg|B|factorial_interaction|+0.311683|[-0.2479490276094931, 0.8713141446766941]|32/32/64/0|
|vidstg|B|Adapted_minus_Native_evidence_fixed_T|+0.298138|[-0.25074673262981767, 0.8470225363052755]|30/34/64/0|

The four-cell interaction is (Full-T)-(S_Native-F). It includes changes to observation positions, candidates, references and Tube-IoU support. It is not proof of optimizer-gradient synergy. The fixed-final-T comparison Full_Adapted minus T_S_Native_evidence uses exactly the same final Temporal interval; dense known-GT spatial IoU also uses the same physical GT frames for all arms. All other scored supports remain explicitly named in JSON.

S-only tIoU equals Frozen and T+S tIoU equals Temporal per source. The complete spatial trajectory is recomputed by the qualified frozen-prefix readout through both native decoder passes. Source analysis establishes that TimeDecoder and action hidden states are independent of the changed PosDecoder parameters and query residual. Existing T head adaptation is an independent sealed branch output. Composing its interval is the original factorized deployment protocol, not a claim of a new jointly trained network forward. Old Adapted-conditioned Spatial readouts are retained with that label and are not renamed true Native-only.

Every per-source signed effect and unknown, paired source-level t interval, reference-admission outcome, actual fit/proposal/acceptance count and observed stage cost is retained. Intervals describe uncertainty for this development cohort; no bootstrap, new selection gate, GT-driven screening, or automatic promotion is performed. Spatial parameter and acceptance mathematics, Source checkpoints, random sampling, visual budget, Prompt and precision remain fixed. Native versus Adapted observation changes are the intended treatment. Semantic512 remains a separate model-assisted pixel review; metric changes do not establish identity/role correctness.

Private media, annotations, GT coordinates, query/Caption/Teacher response text, raw tensors, checkpoints, credentials and conversation records are excluded. No official fullset, CURRENT change or project FINAL.
