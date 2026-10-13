# State-Benefit Retention Audit (one cache-only audit)

The human authorized this audit on 2026-10-13 after the controlled LN state
intervention. All 64 previously exposed P9/P0 previous-P6 parents (32 HC-STVG2,
32 VidSTG), all three Source/Inherited_order1/Inherited_order2 arms, all 192
accepted sealed fits and all 4,800 saved optimization rounds are included.
No model, teacher, forward, backward, training, optimizer execution, new state
intervention, gate, loss, parameter search, manuscript or production promotion
is authorized by this audit. Cancelled baseline processes retain their original
T standby identity. P7-A remains frozen.

## Fixed readouts and offline labels

Read every saved boundary, including boundary zero and the final configured
step: 41 HC and 11 Vid boundaries per arm, 4,992 arm-boundary readouts total.
Before reading labels, seal complete CPU cached readouts and their source SHA
bindings. Labels were already exposed historically; sealing does not create a
fresh confirmation set. Labels only annotate these fixed readouts, never choose
steps, parameters, parents, evidence, or a deployable policy.

Primary Q is parent-equal mean spatial IoU over genuine decoder frames inside
the original half-open GT event, excluding every physically visited Omega0
position (including rejected evidence). Common U additionally excludes Omega1;
observed-event frames are secondary. Keep all real output frames for GT-free
convergence and mark missing/event-excluded GT as unknown. Preserve original
HC lower-bound pixel clipping and Vid unclipped official coordinate convention.
Report full original dense Before/After scores separately from these real-frame
readouts. Anonymous frame ordinals are not physical frame IDs or media.

For inherited arm j, A_k = Q_j,k - Q_Source,k. Erosion is A_K-A_0, exactly
Delta_j-Delta_Source. Separately report absolute inherited change and Source
change. A smaller advantage may reflect Source catch-up while inherited Q
improves. Do not label that event destruction of inherited localization.
Positive initial advantage retention is signed A_K/A_0 only for A_0 >= 0.01;
zero, negative and small initial advantages retain explicit unknown ratios.
All signed advantage transitions and new gains/losses remain in the output.

Frame strata are fixed: initial inherited advantage > 0, >=1pp, >=5pp;
initial inherited IoU >=0.5; and both advantage >=1pp and inherited IoU>=0.5.
Report inherited absolute degradation, advantage erosion, Source catch-up and
their intersections. Persistent degradation means inherited IoU below its own
Before at every boundary in the final quarter (HC30..40, Vid8..10), including
the endpoint. These are offline strata, not online filters.

## Convergence, evidence agreement and gradients

For all three arm pairs and every matched real frame/boundary, report normalized
cxcywh coordinate RMS and box IoU agreement. Keep all-frame, own-U, common-U
and observed-event reductions separate. Output convergence means an actual
decrease in RMS or increase in pair box agreement, not similar mean GT scores
and not parameter/state convergence. Also report joint/QR/LN parameter RMS
separately. No new decoder Jacobian or forward pass is inferred from the cache.

Expert agreement is the original admitted Top1 DINO normalized box IoU at the
same actual visited positions, with the original 1e-7 denominator floor. Read
and independently compare the saved central rewards. No admitted position is
unknown, not zero agreement. Test the parent intersection of expert agreement
increase, advantage erosion, and persistent inherited GT degradation in the
predeclared initially correct/advantaged U stratum. Agreement is not teacher
correctness or a usable actionability predictor.

For each actual update use path[k].update for k -> k+1; read existing gradients,
applied displacements, states and Adam moments. Do not run Adam. Report joint,
QR256 and LN1536 gradient and step norms, pair gradient cosine (zero norms
unknown), cumulative actual displacements and output movement per boundary.
Use every actual round; early quarter is the first ceil(K/4) updates (HC10,
Vid3), and final-quarter endpoints are HC30..40/Vid8..10. No GT best step.

## Statistics, tails and competing interpretations

Parent is the independent unit (32 per target); frame counts are denominators,
not independent sample size. Use paired parent bootstrap 10,000 replicates,
fixed seed 20261013, same parent indices jointly across arm/round metrics.
Intervals are descriptive, unadjusted for multiple comparisons. Preserve all
positive, negative, neutral (1e-12) and unknown outcomes, empty/no-update fits,
both order contrasts, final and early-quarter readouts.

Report leave-one-parent-out mean ranges and diagnostic removal of the largest
one/three absolute erosion parents; never replace the complete primary result.
Describe association of erosion with initial advantage and starting quality,
and leave-one-parent-out linear headroom-model prediction R2. Regression on
starting GT quality does not identify a headroom mechanism or remove the causal
initialization effect; the baseline and change share terms mechanically.

Decision is evidence scoped: convergence, increased expert agreement and
persistent absolute loss on initially correct U frames can support benefit
erosion during current-query sparse adaptation. A signed advantage identity
alone cannot. Mixed/nonconvergent or weak intersection results weaken a single
convergence/overwriting explanation. Neither outcome proves catastrophic
forgetting of previous queries: previous-query retention was not tested.
Do not automatically start a preservation algorithm, gate or full main run.

## Completion

Complete source/state/input/noise/math/frame and independent scalar/parent
statistical readback, actual pixel viewing of every PDF page, exhaustive anonymous
results/code/protocol publication with remote bytes/SHA256/Git blob/mode/tree
checks on both refs, archive check-snapshot-check, and actual ROOT_CLOSING/FINAL.
The same 30-minute monitor remains ACTIVE; this twelfth finite scope does not
complete future method or main-experiment protocol.
