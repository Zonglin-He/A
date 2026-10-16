# Lite actual-clock confirmation closed loop — batch 004

This cohort completed both real Full and Native-clock Student pipelines, independent source/saved-mathematics/readout checks and a joint pre-GT seal before five-arm scoring. Subsequent cohorts are not required for this incremental publication. This is not whole-study FINAL or CURRENT promotion.

## Fixed method and cohort

TA-STVG cross-domain Source checkpoints; 64 sources, 32 per target domain, one query per parent. These64 sources exclude the current128 development parents; earlier historical exposure is disclosed, so they are not claimed globally untouched. No confirmation-driven retuning.

Temporal-Lite: shared candidate supervision multiplied by offset count, no parameter L2, original full-grid per-offset trust and 512-parameter L-BFGS. Spatial: Lite-fullstep/full-grid decoder-response direction update, 1792 Query/LN parameters, episodic reset. HC lr0.0045/sigma0.025/tau0.1/5steps; Vid lr0.027/sigma0.4/tau0.5/3steps. Frozen R1 and existing weight-free candidates remain unchanged. No DINO, YOLO or SAM is added.

The Full branch reconstructs candidates and R1 references at the selected Temporal-Lite interval. Spatial-only uses actual Native-clock references, not Full-clock geometry with a different final interval. Matched_original_OPD uses the same Full references and final parameters as Full, with the original validated OPD update; it is not its separately retuned optimum. No GT enters candidates, references or online step acceptance.

## Complete five-arm result

| Method | HC m_vIoU (%) | VidSTG m_vIoU (%) |
|---|---:|---:|
| Frozen | 22.998670154 | 9.910852445 |
| Temporal | 25.721768841 | 11.514513449 |
| Spatial_native_clock | 22.837174751 | 9.567521964 |
| Full | 25.860137647 | 10.982597174 |
| Matched_original_OPD | 25.870346089 | 10.910654086 |

Full minus Frozen: HC +2.861467 pp; VidSTG +1.071745 pp.

Full minus Temporal: HC +0.138369 pp; VidSTG -0.531916 pp.

Full minus Matched_original_OPD: HC -0.010208 pp; VidSTG +0.071943 pp.

Full improves over Frozen in both domains in this cohort.Its relation to the matched original OPD control is reported separately for each domain above. All negative findings are retained and do not trigger another parameter search.This confirmation result does not retroactively select another method or parameter configuration.

JSON scores are fractions, paired deltas percentage points. Every positive/negative/neutral/unknown source is retained with the saved1e-6 pp neutral tolerance. Native and Full supervision clocks differ: their supervised/unsupervised-frame means do not share identical supports and must not be mixed as a matched-support contrast. Dense GT-frame metrics and complete Tube metrics remain separately labelled. No bootstrap.

Costs contain actual worker elapsed times, including separately labelled failed engineering attempts if present, and Student fit/capture/readout counts. Confirmation Native teacher reuse, if any, must be established by its own actual receipts; development reuse is not assumed. These are not cold-start deployment or official full-set speed measurements. The frozen external expert may carry target-domain prior; overlap-free external training is not established.

Videos, annotations, weights, raw tensor caches, real-query prompts, raw teacher responses and private conversation are excluded. Source snapshots inherit the previously audited complete batch001 closure and add only the necessary new engineering source files.

[All anonymous rows, summaries, paired cases, actual configuration, costs and source bindings](../results/stvg_lite_event_support/2026-10-16/batch004/)

![Five-arm comparison](../results/stvg_lite_event_support/2026-10-16/batch004/FIVE_ARMS.png)
