# Temporal two versus four candidates

Fixed exposed development cohort: HC128 and VidSTG128, one independent parent/query per source. This is not a new confirmation. Frozen and T4 are exact sealed controls. Spatial adaptation is disabled in both Temporal arms.

T2 retains Native and the highest-scoring different legal union of the original two frozen offset grids. Joint score is the sum of frozen grid scores, with original grid-row order for ties. Eight distinct legal physical frames are required; a missing distinct alternative produces an exact Native no-update. No Caption reward, GT, expansion ratio or contraction ratio selects the alternative. The original 512-parameter Temporal-Lite L-BFGS/Armijo, shared-mean candidate KL times offset count, Full-grid Trust, full-grid decode, original Caption/consistency Prompt, 8x448 input, corrected clock, greedy160-token cap and precision are preserved. Only candidate count and the predeclared independent two-candidate eligibility change. Original T4 gate and fallbacks remain untouched.

| Domain | Arm | vIoU % | tIoU % | dense sIoU % | R@0.3 % | R@0.5 % |
|---|---|---:|---:|---:|---:|---:|
|hc2|Frozen|21.651160|39.546835|53.809342|30.468750|7.812500|
|hc2|T4|26.293204|49.042562|53.809342|41.406250|13.281250|
|hc2|T2|26.331915|49.232269|53.809342|41.406250|14.062500|
|vidstg|Frozen|9.843861|32.374371|30.729497|10.156250|1.562500|
|vidstg|T4|10.328918|33.602694|30.729497|10.937500|1.562500|
|vidstg|T2|10.286840|33.278160|30.729497|10.937500|1.562500|

| Domain | Contrast | Metric | Delta pp | Paired source 95% t interval pp | + / - / neutral / unknown |
|---|---|---|---:|---|---|
|hc2|T4_minus_Frozen|m_vIoU|+4.642044|[2.9907004138258935, 6.2933879382533835]|79/23/26/0|
|hc2|T4_minus_Frozen|m_tIoU|+9.495727|[6.650433820910646, 12.34102008799768]|82/21/25/0|
|hc2|T2_minus_Frozen|m_vIoU|+4.680756|[3.1010778893282684, 6.26043341687928]|78/26/24/0|
|hc2|T2_minus_Frozen|m_tIoU|+9.685434|[6.878881006782061, 12.491986915261927]|83/23/22/0|
|hc2|T2_minus_T4|m_vIoU|+0.038711|[-0.6104665243802647, 0.6878894785085372]|37/31/60/0|
|hc2|T2_minus_T4|m_tIoU|+0.189707|[-0.9323354868381832, 1.311749499973843]|38/32/58/0|
|vidstg|T4_minus_Frozen|m_vIoU|+0.485057|[-0.014067239077919236, 0.9841814906119009]|37/25/66/0|
|vidstg|T4_minus_Frozen|m_tIoU|+1.228323|[0.11935380429058351, 2.33729240078034]|46/27/55/0|
|vidstg|T2_minus_Frozen|m_vIoU|+0.442980|[-0.02543153251891561, 0.9113907841462057]|48/26/54/0|
|vidstg|T2_minus_Frozen|m_tIoU|+0.903789|[-0.14820723156951088, 1.9557853537647212]|53/30/45/0|
|vidstg|T2_minus_T4|m_vIoU|-0.042077|[-0.2903109105752213, 0.20615591066852967]|33/23/72/0|
|vidstg|T2_minus_T4|m_tIoU|-0.324534|[-1.0031034841465338, 0.3540354012708207]|39/25/64/0|

All four prelocked effect criteria remain unestablished: every paired95% t lower bound is below -0.2pp. This does not prove T2 is inferior, but the near-zero mean vIoU difference does not establish equivalence or authorize replacing T4. On the pre-adaptation common-eligible subset (HC120/Vid101), T2-T4 vIoU is +0.023652/-0.173166pp and tIoU is +0.145428/-0.561805pp; both domains retain intervals crossing zero. T2 restores two-candidate fit eligibility on35 sources that the original T4 four-way gate skipped. This opportunity difference is disclosed separately from candidate-selection quality.

The all256 table is primary. Pre-adaptation common eligibility is a separately reported diagnostic, with its own domain-specific sample counts; successful updates are never used to select the main cohort. No-information, no-fit and optimizer termination reasons are preserved for every source. Pairwise95% t intervals and signed outcomes include all evaluable sources, with unknowns retained. The prelocked tolerance is -0.2pp for both tIoU and vIoU in each domain; a failed or unresolved criterion does not trigger another search or automatic promotion.

T2 has 512 logical candidate pairs versus T4 884. Actual added work is 35 Caption/alignment pairs, 35 alignment-only calls and 2 extra original-adapter qualification pairs; 442 complete old pairs are reused only after exact Source/input pixels/sampling/FPS/own-query/complete Prompt checks. Separate real stage/per-job costs include cache handling and qualification. They are cached experiment wall times, not a same-process fresh T2/T4 speed benchmark; fresh end-to-end speedup remains unknown.

All predictions and exact full-fit replay are sealed before the new GT scoring pass. GT is absent from candidate selection, Caption rewards, optimization, acceptance and fallback. Existing T4 and failed B confirmation are preserved. No Spatial change, OPD resume, Prompt/LR/budget search, official fullset, CURRENT change or project FINAL. Private query/Caption text, Teacher raw responses, media, annotation/GT coordinates, tensors, weights and credentials are excluded.
