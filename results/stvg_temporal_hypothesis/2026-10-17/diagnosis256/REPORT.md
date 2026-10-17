# Why Native-only Temporal B did not retain T2 gains: original256 CPU diagnosis

Same HC128/Vid128 exposed development parents, sealed B/T2 outputs and unchanged Source boxes. No new model load, GPU job, Teacher call, Student fit, video decode or method promotion. Previously exposed target GT is used only after original prediction roots for these post-hoc diagnostics. This is neither an independent confirmation nor a new fitted B result.

## Matched complete-cohort readouts

|Readout|HC vIoU %|Vid vIoU %|HC tIoU %|Vid tIoU %|
|---|---:|---:|---:|---:|
|Frozen|21.651160|9.843861|39.546835|32.374371|
|Projection|26.947798|10.079683|50.268144|32.831025|
|T2_reward_view|23.412917|9.931525|43.516611|32.656421|
|T2_target_view|24.341237|9.937801|46.108380|32.456573|
|T2|26.331915|10.286840|49.232269|33.278160|
|B_reward_view|21.314999|9.827169|39.187213|32.252322|
|B_target_view|21.522449|9.792448|39.443767|32.214975|
|B|21.518514|9.779438|39.443767|32.181130|

Reward-view chooses the higher saved numerical reward; target-view chooses the higher prior-tilted target probability. Missing, truncated or equal-information cases retain the original method base: Projection for T2, Native for B. These views are diagnostic hard readouts, not replacements for the actual trained Student.

## Projection and the old T2 gain

The source-wise identity is T2-Frozen = (Projection-Frozen) + (T2-Projection). Projection includes the original Actionness normalization, interval cost and 0.1 Native log-prior. It is not a clean intervention on the Actionness tensor alone. The second term jointly includes Teacher feedback and Student updates; it cannot isolate a pure Teacher causal effect.

|Domain|Projection-Frozen vIoU pp|T2-Projection vIoU pp|
|---|---:|---:|
|hc2|5.296639|-0.615883|
|vidstg|0.235823|0.207157|

HC Projection supplies +5.296639pp; subsequent Teacher+Student changes reduce it by .615883pp. Vid gains split into +.235823pp and +.207157pp, both individually uncertain. Projection usually lengthens HC intervals: median Projection/Native duration is 1.342868, so the evidence does not support a story that it simply trims background. No conclusion that all semantic feedback is harmful follows.

## Where B loses useful signal

The fixed B hypotheses have median interval IoU .911213 HC / .947826 Vid and median normalized total endpoint shift .093262 / .054646. Different exact sampled frame IDs (median Jaccard .066667) do not prove different event semantics. The metricwise hard-pair vIoU oracle potential above Frozen is only +1.083200pp / +.629739pp, compared with +6.135457pp / +.864622pp for the old T2 pair. This is a support diagnostic, not an upper bound on free-grid Student adaptation.

The reward ordering agrees with GT interval tIoU ordering for 48/105 HC and 47/92 Vid pairs with informative, non-tied values. Other cases remain tied/unknown. This is geometric ordering, not independently audited event, identity or relation correctness. Raw reward hard-selection itself does not improve Frozen on average. Increasing reward strength or relaxing trust is therefore not the first supported remedy. Thirteen queries have no information (six truncated descriptions, seven equal finite rewards; zero empty descriptions).

## Student transfer versus optimization convergence

Final B intervals match the current prior-tilted target hard readout on 253/256 sources (126 HC, 127 Vid); all B final intervals are inside its fixed pair. Actual B minus target hard readout is -.003936pp / -.013010pp. Median realized candidate log-odds shift is .773212 / .760359 of the requested shift; direction agrees on 123/123 HC and 119/120 Vid informative fits. The single contrary Vid reward difference is approximately 3.7e-9, so its ratio is numerically unstable.

Candidate KL falls from .008270 to .000524 HC and .008720 to .000597 Vid. This is transfer evidence, not full convergence: most original fits hit 50 iterations; final median gradient max is 1.59e-4 / 1.25e-4, above the original 1e-10 tolerance. Student limitations may remain, but they do not explain the main HC deficit when the current target hard readout has almost the same task score. Old T2 decodes outside its observed pair on 73 HC and 69 Vid sources, illustrating its different projected full-grid path.

## A no-Actionness probability/readout probe

Current B retains one latent two-offset state per physical interval. That state has median only 6.648177% / 5.844341% of the total Native probability assigned to the same observable merged interval. This was an explicitly declared representative-state approximation, not a falsely claimed marginal distribution.

A single rule was locked before new probe scores: push forward the independent offset posterior by summing probabilities of all latent states merging to the same physical interval, then choose the physical interval MAP. No model, Teacher, fitting or parameter search is involved. MAP after summation differs from merging two separately chosen MAP states. All256 probability tables were independently reconstructed with a two-dimensional CDF/inclusion-exclusion implementation; no probability was clamped.

```text
P_N(I) = sum_{z: merge(z)=I} product_o p_N^o(z_o)
I_MAP = argmax_I P_N(I)
```

|CPU-only readout|HC vIoU %|Vid vIoU %|HC tIoU %|Vid tIoU %|
|---|---:|---:|---:|---:|
|Native merged-posterior MAP|24.945094|10.299485|46.626294|34.126993|

Versus Frozen: +3.293934pp HC, paired source t95 [1.845394,4.742474], 79+/25-/24neutral; +.455624pp Vid, t95 [-.133778,1.045027], 48+/32-/48neutral. Versus T2: -1.386821pp HC, t95 [-2.595696,-.177947], and +.012645pp Vid, t95 [-.335514,.360803]. These exposed-development findings are a new parameter-free CPU readout, not an improved fitted B result, independent test or established Vid gain.

The same prelocked probability ranking supplies an alternative to the original Native interval with distinct actual uniform8 samples on all256 sources. Offline hard-pair oracle potential becomes +4.592522pp HC / +1.086232pp Vid; no new pixels, descriptions, rewards or Student predictions have been generated for this new support.

## Next minimal discriminating test (proposal only; not launched)

1. Keep the current Native I1, Teacher/Prompt, eight-frame budget, reward, temperature, trainable512 parameters, optimizer and Native trust unchanged. Replace only the I2 ranking by observable merged-interval Native probability; keep its representative-state training semantics explicit. This isolates support construction, without Actionness, Expand/Contract or an optimizer search.
2. Retain the measured parameter-free marginal MAP readout as a separate no-update control. Its free decoder gain must never be credited to Teacher supervision or TTA. Acquire real new-I2 visual evidence and fresh feedback before fitting; do not reuse unrelated old captions for new intervals. Seal all predictions before evaluation.
3. Report candidate support potential, numerical Teacher ordering and realized Student metrics separately. If same-Teacher ordering remains unhelpful despite improved support, evaluate a separately locked pairwise semantic comparison later; do not simultaneously alter Prompt, reward scale and optimizer.
4. A fully marginal candidate-target objective would require log-sum-exp over all states representing each interval, with corresponding gradients; it is a distinct future representation change, not a silent rescaling of the current joint-state loss.

No new GPU experiment or rollout was launched by this diagnosis. Original negative B, failed confirmation, CURRENT/PAPER and other queues are preserved. Native posterior aggregation currently uses TA-STVG two-offset structure; cross-backbone plug-and-play has not been measured. All original source rows, signed positive/negative/neutral/unknown outcomes, implementation limitations and post-hoc data exposure are retained in the accompanying JSON.

