# Why Strong448+224 beats current Global6: matched saved-evidence analysis

This is a read-only analysis of the same already scored HC64, not a new accuracy experiment. No GT file, video, model/head, GPU, decoder, expert or optimizer is used. All64 content-bound captures are hash checked only for ERef size/grid metadata. Data have repeated development exposure.

## First correct the comparison

Actual source `stvg/predictor.py` defines qStrong=.5*q448,last+.5*qGlobal6, where qGlobal6=mean over six224 layer posteriors. Both use the same original two offsets, physical grid, offset hull and exact MBR. Global6 already contains the224 six-layer average; the old statement that Strong versus Global6 also changes layer fusion was inaccurate and is corrected in docs/temporal_audit.md and temporal_sources.md.

The conditional effect of **adding the448 posterior to the fixed224 six-layer posterior** is measured. This does not identify448-only accuracy, nor separate finer detail, query-initialization changes and multiscale calibration.448 does not add temporal frames or denser physical endpoints.

| Same BoxFixed spatial implementation | tIoU % | Full vIoU % | Boundary MAE s |
|---|---:|---:|---:|
| Native224 last layer |54.8230|29.0568|2.6821|
| Global6,224 six layers|55.9613|29.6952|2.6034|
| Strong,448 last + Global6|58.6043|31.3985|2.5113|

Strong−Global6 t+2.6430pp [0.4436,4.9534], full v+1.7033pp [.3634,3.1196]. Fix the exact same Strong spatial track for both intervals: v remains+1.5864pp [.2256,3.0129], with spatial scores identical. Thus most measured full-v gap survives pure time replacement; interval-conditioned final DP contributes a smaller .1170pp residual in this particular decomposition. It is not a claim that vIoU generally factorizes into independent time and spatial averages.

## Mechanism evidence versus hypotheses

1. **Upstream input differs.**224 cached TimeDecoder grids are7×9 or7×10.448 recomputes appearance/motion features, multimodal encoding, original native query construction and temporal decoding at the new scale. It is not merely upsampling old224 features or reweighting a final attention map. Current tube replay changes the final decoder's attention under already cached224 memory/query; generic box-consensus readout changes only the final ranking. Neither reconstructs evidence discarded upstream.
2. **Training-scale proximity is plausible, not established causality.** Original paper and official yaml train with420 image scale.448 is numerically closer than224, but our width cap/aspect rule and source-domain transfer remain; no matched420/448/224 ablation is claimed. Small hand/object interactions or motion states may benefit from finer spatial encoding, yet this audit does not visually establish that as the main cause.
3. **Six layers are not six independent visual observations.** Their inputs and backbone are shared. Their mean can change calibration and stabilize some layer-specific errors, but it cannot independently recover absent visual distinctions. Here native→Global6 gains1.1383pp t, while adding448 gains2.6430pp. These are measured sequential differences, not a universal causal percentage decomposition.
4. **Posterior fusion can hedge errors.** Different resolutions change the entire frozen inference trajectory. Same-interval grids make posterior averaging meaningful; the final expected-IoU decision can change even if each input's own MBR differs. Whether448 provides better likelihood or complementary calibration is unmeasured without a matched448-only posterior audit.
5. **Current spatial feedback lacks event discrimination.** A full target box can persist before/after the queried action. Spatial agreement can improve target localization without identifying its action start/end. Previous true/permuted replay and current generic-consensus controls failed to supply reliable new event evidence. This is a result about tested feedback designs, not impossibility of spatiotemporal reasoning.

Original architecture and training source: [TA-STVG paper](https://arxiv.org/html/2502.11168v1), sections3.1–3.4 and4.1; official local source `external/TA-STVG/models/pipeline.py`, native temporal preservation `stvg/temporal_capture.py`, input transform `stvg/inputs.py`. No internal actionness score is introduced as our new method; this discussion audits why the original frozen backbone's full inference differs.

## Actual cases, not just an appealing explanation

Strong improves36/64 tIoU, harms22, ties6. It rescues4 queries across tIoU .5 and harms2 formerly above.5. So it is better on average, not universally safer.

| Anonymous query | Global6 tIoU | Strong tIoU | Concrete boundary change |
|---|---:|---:|---|
| H023 |.3133|.5863|End absolute error6.7887→3.6606s; start1.9634→1.3644s|
| H058 |.6886|.9179|Same start error .5324s; end2.2962→.2329s|
| H051 |.8967|.7329|Start error .1597→2.3154s; interval10.6587→13.4531s|

For H051, saved predicted interval length10.6587→13.4531s, while true event length9.8603s. This is a counterexample to uniformly expanding boundaries. Strong's average interval is .5052s longer but its length absolute error is .2203s worse; the latter CI crosses0. Better tIoU does not mean every boundary or duration statistic improves.

Descriptive posthoc grouping: <5s events13 queries, t gain+1.0067pp;5–10s37 queries,+1.9522pp;>=10s14 queries,+5.9878pp. These data do not support explaining the result only by tiny/brief events. ERef full-box area versus t gain has Spearman .0107; box extent is not the size of a discriminative hand/object part, so this neither proves nor excludes a detail mechanism.

## What the224 candidate pool does and does not explain

Against the GT diagnostic best of the fixed13 existing224 decisions: Strong is better in22 queries, equal3, worse39. In one Global6-wrong query, Strong crosses.5 while none of the13 candidates can. Thus merely reranking this fixed13 bank cannot reproduce every Strong success. This is a finite-candidate limitation, not proof that all224 hidden/posterior support lacks the answer.

Conversely the224 candidate oracle averages5.0366pp t above Strong, but requires GT selection and is not deployable. Therefore two bottlenecks coexist: useful alternative information/decisions from the448 route, and failures to choose already available224 alternatives. Current uncalibrated spatial signals have not solved the latter.

## Consequence for a light method

Keeping quality while removing448 requires recovering its useful *task-relevant distinctions*, or using a limited high-resolution computation where that is justified. More weighting of the same spatial consensus is not demonstrated to do this. A future efficient design must separately establish (a) useful missing event evidence, (b) correction of already available candidates, and(c) real backbone/IO cost. An entropy-only high-resolution gate is not currently justified: confident wrong predictions were a documented earlier failure.

No resolution/gating/weight experiment or full-run change is authorized by this analysis. The active full evaluation continues unchanged. HC64 matched development results must not be compared directly with HC1902 full averages to claim a method-level drop or overfit.

All64 saved per-query metrics, signed changes, confidence intervals, controls and source hashes are in analysis.json. No query text, video, raw frame boundaries, boxes, features or weights are published.
