# Evidence-Conditioned STVG: bounded P0/P1 experiment

Human authority: the complete 2026-10-15 attachment b71aefb6. The earlier truncated attachment is superseded. Current deployment registry is protected. Old 13,785-query run stays paused.

## Actual starting evidence

The corrected development32 winner was Offset-Trust + new-clock crop (HC/Vid vIoU 34.643557/10.784159%). The previously evaluated independent32 subsequently measured F/T/S/TS as HC 22.612489/24.330399/22.272852/24.000041 and Vid 20.792291/21.410192/21.250628/21.986363%. It is not an untouched set for further design. No independent performance claim follows from reusing development32.

## Mechanisms and matched controls

P0 uses saved actual per-round antithetic actions, rewards and full decoder outputs from development32. Independently check novelty, population variance, overlap, ESS, reference fitting and parameter displacement. Reward support is not a semantic correctness certificate or proof of useful gradients. The existing code already centers weights by 1/32, skips exactly flat reward policies and uses external reliability-weighted per-anchor loss; these are not new repairs.

P1 holds candidate geometry, input NPYs, four center clocks, Source state, offset-trust interval, five steps, 32 antithetic samples, 1,792 active parameters and the already locked domain-specific config fixed. A/B: neutral identification versus student-conditioned correction, both with identical KEEP/REPLACE/ABSTAIN options, closed labels, video budgets and output-token cap. C/D pair these same reference sets with evidence-aware rather than original validated OPD. Historical top-1 is reported separately because its output task did not include the new matched response schema.

At most 256 main R1 generations and 64 preselected ID-permutation/visual-null controls. Controls are first four aliases/domain and center slots 0 and 2, without looking at GT. Gray visual-null retains the ID strips and clocks, removing the contact-sheet image content; it does not establish which referent is correct. ID invariance is diagnostic and is not silently made a hard gate. Malformed output remains an invalid observation; no prompt retries selected by scores.

Evidence-aware weights: reliability times source/reference geometric novelty times closed-label evidence gate, normalized outside the per-anchor objective. Keep an anchor only when its reward population variance exceeds 1e-12, at least one of 32 sampled IoUs exceeds 1e-6, and novelty exceeds 1e-6. Closed semantic labels are uncalibrated Teacher assertions. All zero admission gives no Adam step. Existing weighted reference-IoU nondecrease acceptance and full optimizer rollback remain; that imperfect-teacher validation cannot guarantee true tube improvement.

Every case retains F, T, S/TS for A–D and historical S/TS. Record invoked fits separately from fits with accepted nonzero updates and exact reuse. Actual full Student decoder outputs must match the fitted replay and preserve frozen source weights. Keep all unknowns and negative cases in the 32-source denominator. No GT/model oracle fits. Only after all predictions and independent root readback are sealed may the separate scorer read GT. Report query/source means, positive/negative/neutral sources, worst cases, tIoU, dense and fixed-observation/non-observation sIoU, plus entire end-to-end cost. No bootstrap.

## Deferred experiments and stopping

Cold start is disabled, not a default fourth module. It requires evidence of supported/corrective references with poor action support across multiple sources and a separately locked matched comparison. Student-aware 3-global/3-local/2-temporal proposals are a separate budget-matched followup only if candidate coverage is established as limiting. Boundary-contrastive temporal feedback is separate from P1; no simultaneous time-reward change. Neither is implemented as an untested always-on main method. No new teacher/detector/tracker, broad hyperparameter search, or production promotion.

Selection uses the complete exposed-development results with one globally chosen recipe, not dataset-wise A/B/C/D picking. A freshly registered confirmation set cannot select prompts or gates. The old full run is not resumed by this bounded controller. A new full run requires frozen supported recipe and confirmation, even if confirmation is below the old eight-case peak; negative findings remain reported.

## Source notes

Primary materials checked this round: [Rethinking OPD](https://arxiv.org/abs/2604.13016), [official code](https://github.com/Thinking-Space/Rethinking-OPD), [TubeRMC](https://arxiv.org/abs/2511.10241), [Decomposed OPD](https://arxiv.org/abs/2606.00564), [BAM-DETR](https://arxiv.org/abs/2312.00083), [STVG-R1 official code](https://github.com/XiaowZhang00/stvg-r1). Abstract/official README verification is not an independent reproduction or a claim to have audited all appendices. LLM token-level results are hypotheses for continuous box policy adaptation, not established box-OPD theorems.

The existing proposal wrapper calls OpenCV Selective Search/GrabCut/DIS APIs; this is not a new segmentation algorithm. Installed opencv-python wrapper notices are MIT and the bundled OpenCV library notice is Apache-2.0 with third-party notices, retained in the original dependency package. No detector, wheel, checkpoint or private dataset is redistributed by these project-source changes.
