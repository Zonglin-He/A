# Large-correction temporal evidence audit

Authorized by the user's 2026-10-03 inline follow-up to ac2102e. Question:
can existing, non-PE boundary signals distinguish beneficial large endpoint
changes from harmful ones on the **same** sealed A8 / Expanded32 supports?
This is a bounded CPU evidence audit, not a new deployed refiner or expert.

## Inputs and fixed conditions

Use the exact preceding 1,152 arrivals: each dataset's 32 development and 16
confirmation sources, one query/source, two orders, clean + five 5% transient
corruptions, 25% expert arrivals. All sources have historical exposure.
Only the 288 expert arrivals (240 corrupt, 48 clean) have 32 hypotheses.
864 nonexperts retain A exactly; no new candidate generation for them.
Old8 is the complete prefix of Expanded32; original A8 is the anchor and remains
available. A spatial boxes, Uniform Rank-RKL persistent transitions, checkpoint,
pixel hashes, original Paper48 sampling, expert locations, Vid K1 and HC K8
parameters all remain frozen. No spatial score, parameter adaptation, memory,
third view, PE curve, expert invocation, media or weight access.

Availability limitation: the online A logs did not retain its full temporal
logits. N below uses **frozen source-checkpoint** logits on the identical input,
not A-current logits. Report native interval agreement with A's pre-Fast native
interval and with A8 separately. Do not claim a current-state likelihood test.
The other two signals share the existing UVTG model; another sampling view is
not an independent expert. Statistical independence is not assumed.

## Label-free signal construction (locked before joining candidate GT metrics)

N: normalize start/end logits within each of the original two offsets,
give each offset probability mass 1/2 and interleave on the physical merged
grid. Score I=[f_i,f_j+1) by log p_start(i) + log p_end(j).
This is a marginal boundary-prior proxy, **not** the native two-offset-envelope
joint probability; it allows cross-offset endpoints. Use float64 log-softmax
directly, without exponential underflow. No smoothing or temperature search.

U: use every cached UVTG proposal boundary with equal mass and no confidence.
At each candidate start/end compute Gaussian kernel density of the corresponding
proposal starts/ends. Bandwidth is **0.5 seconds**, the existing 2Hz expert
sampling period, fixed for both datasets and views, not fitted to GT. Score is
log K_start(s) + log K_end(e). Compute log-mean-exp stably. Keep duplicate and
out-of-window proposals; do not merge intervals, weight by confidence, or pair
a candidate with proposals by interval IoU. Marginals can combine incompatible
proposal endpoints, a stated limitation. Empty proposal evidence falls back to A.

S: repeat that exact boundary-density calculation on the already captured real
phase-.25-second view. Compare each view's total log score against the same
fixed A8 interval, then S(I)=min(U0(I)-U0(A8), U1(I)-U1(A8)). This is a
**two-view robustness diagnostic of boundary density**, not a third expert,
not a new claim about independence, and not the prior confidence-times-IoU
min-view rule renamed. Publish both view scores and differences so that
agreement, disagreement, and shared mistakes can be inspected.

For all three signals and both supports, choose a unique maximum strictly
above A8's score (tolerance 1e-12); otherwise retain A8. Any top tie keeps A8.
This fixed anchor-relative diagnostic readout tests whether score preference
can safely authorize a change; no thresholds, mixtures or parameters are fitted.
Seal all scores, geometry, decisions and input bindings before loading the
previously published candidate vIoU/tIoU. Labels are cached official-dense
metrics, not new annotation reads. This is nevertheless GT-derived diagnosis,
not an untouched test, GT-free evaluation or final-method promotion.

## Geometry and primary diagnostic

Let ds=s-sA, de=e-eA, r=(|ds|+|de|)/(eA-sA), and
b=2 min(|ds|,|de|)/(|ds|+|de|). Large means r>=.5 (the previous audit radius);
endpoint dominance means b<=.25. These are descriptive fixed conventions,
not learned acceptance gates. With dominant start, ds>0 is trim-start and ds<0
is expand; with dominant end, de<0 is trim-end and de>0 is expand.
For balanced changes, outward endpoints are expand, same-sign endpoints shift,
and inward endpoints **trim-both**. Preserve trim-both and same as explicit
additional categories rather than forcing every change into the four requested
groups. Publish continuous displacements/balance and exact endpoint directions.

Primary sets: all large changes, and large trim-start / trim-end separately;
also all changed candidates, expand, shift, trim-both as diagnostic controls.
Assess candidate benefit/harm by cached dense vIoU relative to the **same A8**;
zero changes are recorded and excluded from binary discrimination, not called
correct. Positive evidence is strictly score(I)>score(A8). Report TP/FP/FN/TN,
precision, beneficial-change recall, harmful-change acceptance, and within-cell
benefit-versus-harm AUC (score ties get .5), on identical candidate populations
for N/U/S. Do not compare different pair populations as calibration proof.
Absent positive/negative classes make AUC undefined, never .5 or zero.

Report anchor-relative selected vIoU/tIoU, same-support oracle regret,
gross gain/loss, >5pp negative tail, .3/.5 correct-to-wrong and wrong-to-correct,
old beneficial Fast results destroyed, source influence and both orders.
Corrupt expert is primary; full-flow dilution and clean are reported separately.
No pooling Vid/HC into an unqualified claim; HC confirmation remains secondary.

## Aggregation, scope and delivery

Average candidates within each arrival for discrimination numerators and
denominators, then conditions, orders and sources. Paired source bootstrap
10,000 draws, seed 20261003, shared draws for ratio numerator/denominator;
report undefined and zero-denominator draws. AUC is conditional on cells with
both classes; publish eligibility counts. Individual candidate pairs are not
independent experimental units. All three signals are exploratory tests with
no multiplicity-adjusted confirmatory claim; confirmation never selects a rule.

If a signal associates with useful large corrections but readout remains harmful,
association does not establish a selector. If cached signals fail, retain A and
report the missing evidence rather than adding an expert automatically.
Frozen-N failure cannot rule out A-current logits, joint span models or trained
quality heads. No extra frozen expert, new model, refiner or full-query job is
started. Publish code, fixed rules, anonymous signal arrays, labels already in
the predecessor export, all summaries/cases/figures and limitations; verify
Zonglin-He/A bytes and archive check/snapshot/check before closing.
