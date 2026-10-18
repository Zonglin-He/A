# One source-calibrated DINO-support supplement to frozen joint correction C

Explicit human scope: one NEW output group E on original exposed HC16/Vid16.
Reuse all frozen A/B/C/D outputs and scores. No new network training, GPU, expert,
media or target data; no dense OPD, raw900 expansion, new admission or production
promotion. Same Before primary/Source sensitivity, original admitted56/50 frames,
168/144 retained candidates, four-param(cx,cy,logw,logh), whole-tube scope and MBR.

## Exactly one added information term

Same-frame relative DINO cost l_ik=log(max_j(max(s_ij,1e-12)))-log(max(s_ik,1e-12)).
This fixed monotone transform is defined before source calibration. The numerical
floor prevents log0 and is not a candidate filter. Every old low-score box remains.
Scores are neither calibrated identity probabilities nor geometric variances.

E(delta)=sum_i min{lambda, min_k[d_ik^2 + beta*lambda*l_ik]}.
The old source scale Sigma and cutoff lambda are BIT-IDENTICAL to C. Dividing this
energy by lambda gives unit outlier cost and a dimensionless shared beta. Thus
different old lambda units are respected, without a separate target-domain beta.
No change to the outlier penalty, no Student-distance prior, no new query gate.
Exact finite <=4^4 assignment enumeration and conditional LS means, deterministic
discrete tie breaks, identical global application. beta0 must exactly replay C.

## Source-only, finite, prelocked calibration

Original source-train72 caches are used ONLY for transform diagnostics and checking
score range/old pool/winner integrity; they do not retune old scales or train a net.
Use original source-validation12/domain (24 total) ONLY to select ONE beta shared
across both source domains. These are exposed development source-validation sets,
not a new independent performance estimate. All source/target parents remain disjoint.
The predeclared positive grid is {1/16,1/8,1/4,1/2,1,2}; beta0 is a reference control
but not an eligible supplement (otherwise the sole new-information test disappears).
Use each source domain's existing C scale/cutoff. For each beta generate all24 source
outputs before evaluating them. Selection maximizes the equal-domain/equal-query
mean change of known unobserved ACTUAL decoder-frame sIoU versus that source's
frozen Native trajectory. Exclude all original planned4 locations, not only admitted.
Source Native time is held fixed for secondary full-vIoU: source MBR is not cached
and no temporal readout is invented. Exact tie: higher pooled full-v change, then
smaller beta. Empty-reference queries remain no-op/all denominators. If an entire
source domain has no evaluable unobserved support, stop; do not substitute target.
No per-domain beta selection, target metric access, temperature, kernel, floor or
grid extension. Publish all source grid results, including beta0 and negative results.

## Target barrier and the attribution requested by the human

Lock selected beta/source results and source inputs before generating any E target
output. Only E is a new formal method group. Old target GT/identity labels and old
target scores are not read by the generation function. All32 E outputs in both
fixed bases seal before any new target evaluation.

Predeclare two-factor OUTPUT diagnostics (not selectable method groups) to separate
candidate changes from accepted-frame changes. At the final C/E delta compute each
frame's best candidate index BEFORE its outlier test; latent index remains defined
even for rejected frames. Combine candidate-index choice (C or E) with accepted
frame mask (C or E), fit their conditional arithmetic mean and globally apply it.
The two endpoints exactly reconstruct C and E. All four diagnostic outputs seal
alongside E before GT. Postseal finite two-factor Shapley values split E-C endpoint
changes between candidate identity and frame acceptance, averaging both replacement
orders. This is a convention-dependent algebraic attribution with interactions,
not independently identified semantic/causal effects. Rejected latent candidates
are not actual accepted references. Never choose a hybrid as a winner.

## Evaluation and fixed exit

Reuse old A/B/C/D metrics and record their exact hash. New E official v/t/s/R03/R05,
known observed/unobserved/within-MBR/interpolated support, per-parent paired changes,
10000 bootstrap seed20261018, serious<-5pp tails, saved-score leave-one sensitivity,
candidate quality and abstention/selection counts. Unknown GT stays unknown and all
16/domain remain. Compare E to Base/A/B/C/D. Relative score/energy reduction is not
a task metric. Use old conservative Top1 model-assisted labels only after seal;
never transfer its identity label onto alternative boxes or infer identity from IoU.

Mean screen for a future separately locked confirmation: E-Base fullv/unobserved
positive in both domains; HC E-A positive on both (retain candidate benefit); Vid
E-A nonnegative on both (restore A gain), with numerical tolerance1e-10pp. Report
tail/CI uncertainty separately; do not require all CIs significant or conceal harm.
If fail, stop this family of simple geometry+DINO-score combination rules: no new
score transform, beta/grid extension, kernel, gate or dense OPD. If pass, only a
confirmation protocol may be proposed; no automatic expert/model run or dense OPD
is launched by this cached32 script. No domain-wise C/E/D recipe or CURRENT change.

CPU<=1hour, no GPU/model lease. Required closure: independent solver/source-choice/
output/metric/attribution audit, all signed anonymous outcomes, actual aggregate
figure inspection, GitHub exact-content dual-ref CAS verification and archive
check/snapshot/check. Preserve all original pins, predictions, failures and user work.
