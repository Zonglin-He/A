# Why Query Consistency currently adds no established Temporal gain

Same original exposed HC128/Vid128 and sealed B2/B3 outputs. This is a postseal CPU diagnosis of existing feedback and parameter states: zero new model/Teacher/Caption/Student/GPU/video calls. The previously exposed GT is used only for offline interval ordering and diagnostic readout scores. No parameter, prior, reward, threshold or candidate is chosen using this diagnosis online. Query Consistency remains the research core; exact marginal MAP is shared final readout.

## Fixed diagnostic readouts

Hard views below are postseal diagnostic selections from saved evidence, not new fitted methods. Invalid/equal-information cases retain Source marginal no-update, all256 retained. Source pair prior is shown separately.

|Domain|Readout|vIoU %|tIoU %|
|---|---|---:|---:|
|hc2|Frozen|21.651160|39.546835|
|hc2|Source marginal no-update|24.945094|46.626294|
|hc2|Native pair-prior hard view|24.945094|46.626294|
|hc2|Raw Query-Consistency reward hard view|23.281513|43.365640|
|hc2|Prior-tilted target hard view|24.859785|46.620836|
|hc2|Actual B3 adapted Student|24.832876|46.553780|
|vidstg|Frozen|9.843861|32.374371|
|vidstg|Source marginal no-update|10.299485|34.126993|
|vidstg|Native pair-prior hard view|10.303761|34.143977|
|vidstg|Raw Query-Consistency reward hard view|9.905904|32.963980|
|vidstg|Prior-tilted target hard view|10.292677|34.114635|
|vidstg|Actual B3 adapted Student|10.290674|34.144786|

|Domain|Contrast to Source marginal no-update|Delta vIoU pp|paired source t95 pp|+/-/neutral/unknown|
|---|---|---:|---|---|
|hc2|Native_pair_hard|+0.000000|[0.0, 0.0]|0/0/128/0|
|hc2|QC_raw_hard|-1.663581|[-2.796734084829862, -0.5304270915614775]|13/39/76/0|
|hc2|QC_target_hard|-0.085309|[-0.3365578914951786, 0.16594037118623245]|7/4/117/0|
|hc2|B_MARGINAL_DECODE|-0.112217|[-0.3711482686246771, 0.1467133348660843]|17/12/99/0|
|vidstg|Native_pair_hard|+0.004276|[-0.0041852053840052104, 0.012736741669743989]|1/0/127/0|
|vidstg|QC_raw_hard|-0.393581|[-0.7715871061748389, -0.015575040280794428]|19/26/83/0|
|vidstg|QC_target_hard|-0.006808|[-0.09320864443242799, 0.07959193240444506]|6/5/117/0|
|vidstg|B_MARGINAL_DECODE|-0.008811|[-0.12590347414325234, 0.1082818533609819]|11/8/109/0|

## Teacher feedback: direction is the first bottleneck

All512 saved captions are nonempty; this is not the earlier missing-caption input bug. HC has120 informative queries/8 no-information; Vid119/9. One HC and eight Vid caption outputs hit the160-token cap (caption count, not query count); equal finite rewards occur on7 HC/2 Vid queries. Some truncations occur twice in a query; these counts must not be added as independent sources. There are4 HC/1 Vid exact duplicate caption pairs.

A/B/C allowed-token total mass is high (means .998794 HC/.998729 Vid, minima .995890/.995426). The first-token wire/normalization does not show an omitted-label or missing-parser failure here. Conditional support posterior means are only .039988/.104616, uncertainty means .620125/.611161. These are model probabilities, not calibrated semantic accuracy. Neither uncertainty nor missing mention automatically proves contradiction.

Among informative pairs with unequal GT interval tIoU, reward ordering agrees on53/110 HC and41/95 Vid. This is not independent semantic accuracy: a caption may support the event while covering a loose time interval. Disagreement can arise from caption omissions/hallucinations, independent entailment scoring, or mismatch between event existence and tight temporal localization. It does not alone prove performance below chance.

Raw reward hard selection loses1.663581pp HC (t95 [-2.796734,-.530427]) and .393581pp Vid ([-.771587,-.015575]) versus the strong same-decoder control. Thus simply increasing reward influence or weakening the Native prior is not currently supported. The measured Native prior is providing protection against this saved feedback on average. Nominal source t95 intervals are not multiplicity-adjusted.

Six predefined postseal representative text cases were inspected: first informative case/domain plus each domain largest positive and negative B3-minus-control. Generic motion can omit required multi-event structure; independent support-minus-contradiction can rank relative event coverage poorly; a high support posterior can lack explicit key-action coverage in the generated text. These are text-only observations, not pixel-verified identity/action labels or a new semantic panel. Raw Query/Caption text is private.

## Target construction: weak feedback mostly perturbs an existing strong choice

Requested candidate log-odds shift equals reward difference/tau exactly; all256 saved tilted targets independently reconstruct. Among informative queries, absolute reward-gap medians .101078/.069116 imply median absolute conditional probability shifts .022233/.017208. The prior median absolute log-odds is about .369650/.226781 on informative pairs. Prior pair hard choice is I2 for119/128 HC and111/128 Vid; feedback changes the target hard winner on only12 informative queries in each domain. This describes limited influence, not proof the prior must be weakened.

Source marginal MAP is already in the observed pair for128/128 HC and127/128 Vid. The pair-plus-control metricwise oracle headroom averages+1.298587pp HC/+.630607pp Vid; much of the cohort has no hard-pair improvement opportunity over the stronger control. This is a hindsight support diagnostic, not an online selector, an achievable Teacher result, or an upper bound on free-grid Student adaptation.

## Student: updates are active and mostly follow the requested direction

All239 informative Student updates move conditional log-odds in the requested direction, both in the float64 fit and actual FP32 head outputs. Stable requested-shift completion medians are .846731 HC/.856034 Vid, with a separately disclosed abs(reward gap)>1e-6 numerical-ratio diagnostic (not a new inference threshold). Candidate KL means fall .004863->.000176 HC and .011670->.000456 Vid. Actual FP32 versus training candidate-probability maximum difference is1.4161e-7; saved objective reconstruction error is0. There is no evidence here that gradients are disconnected or the update was never executed.

Optimization is not certified converged:238 queries reach50 iterations, one stops on Armijo rejection,17 have no information. Final gradient-max medians including no-information sources are1.4965e-4/1.4368e-4, above the original1e-10 tolerance. Nevertheless the prior-tilted hard target itself has no positive mean increment over the control (-.085309/-.006808pp), close to actual B3 (-.112217/-.008811pp). A hard target view is not an upper bound on the Student; this small mean residual does not prove optimization irrelevant, but offers little reason to prioritize stronger fitting before improving feedback.

The candidate objective constrains the ratio of two interval masses while full-grid trust constrains the whole offsets. Their summed probability is only about2.6% on average and changes little; low total probability alone is not a failure criterion. The final adapted MAP lies outside the observed pair on24 HC/15 Vid queries, exposing an additional transfer/coverage limitation. Do not interpret lower KL as task correctness.

## Where positive and negative contributions cancel

The following is GT-posthoc geometric-order stratification, not an online gate or a causal semantic accuracy experiment. Contributions use the full128-source denominator.

|Domain|Reward agrees with GT tIoU ordering|Reward disagrees|Tied/invalid|Total B3-control pp|
|---|---:|---:|---:|---:|
|hc2|+0.153531|-0.265749|+0.000000|-0.112217|
|vidstg|+0.056817|-0.065628|+0.000000|-0.008811|

Correctly ordered subsets contain real positive cases. Incorrectly ordered subsets offset those means; the data do not imply Query Consistency is universally impossible. Nor may these GT-defined subsets be used to keep only successful queries.

## Minimal no-Actionness next candidate (proposal only, not run)

First test relative Query Consistency from the SAME two saved captions under the same original Query, instead of subtracting two independently obtained support-minus-contradiction scores. Ask which description more completely covers the required event order and roles, allowing equality/insufficient information; do not assume either interval is correct or impose shorter-is-better. This changes the feedback task, not the decoder, hypotheses, Native prior, Student optimizer or trust. It cannot recover visual information absent from both captions and has no promised benefit.

Reuse all512 captions: zero new video/caption generation. A future finite Teacher-only check on the same256 would use both presentation orders (at most512 text prefills), restore interval identities and expose order sensitivity/abstention. Calls may match the old independent scoring count, but token/wall costs are not assumed equal. Keep outputs sealed before diagnostic GT, retain all negative/unknown cases. Student reward/target mapping must be locked without fitting target GT before any later adaptation trial. No such call, fit or queue is launched by this diagnosis. Do not simultaneously increase reward strength, change tau/trust, add another hypothesis, restore Actionness or add global/Spatial modules.

Pairwise judging is itself susceptible to position bias, so the order-control requirement is based on prior evaluation research, not evidence this new variant is better: Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, https://arxiv.org/abs/2306.05685 . Distinguishing entailment/contradiction/neutral also remains essential: https://nlp.stanford.edu/projects/snli/ . These references motivate checks, not a new STVG efficacy claim.

## Verification and limits

Independent saved scalar/source-sign/unknown/t95 checks 118464; original postseal baseline checks 4608, full saved baseline equality checks 14336;512 caption records and256 target reconstructions. All initial/final fit hashes remain bound; no model, Teacher, fitting, media decode or production change. Previously exposed development GT only used for this postseal diagnosis; no independent confirmation or new semantic labels. Teacher training-domain overlap remains unresolved. CURRENT/PAPER and historical negative results remain. The512-window pixel-semantic panel is not completed here.


The informative flag means a finite non-equal admitted reward, not independently verified semantic information. Public paired tables retain eight diagnostic pairs; complete12-arm source metrics and every source detail remain available for arbitrary contrast reconstruction. Full local pairwise audit is preserved.
