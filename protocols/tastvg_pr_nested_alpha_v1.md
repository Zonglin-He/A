# Nested source-level regularization audit

User authorization: attachment b21ac9b2 after the completed source-LOSO audit,
GitHub bf45294c2de6cce96ce808da080ecb79dfe4298f. Only alpha selection changes.
This is supervised CPU diagnosis, not label-free test-time adaptation.

## Fixed setting and population

Reuse exactly the predecessor's 192 search expert cells: VidSTG 96 cells from
16 independent sources; HC-STVG-v2 96 from 14. Per dataset: 80 corruption and
16 clean cells. Original design: 32 search sources, one query/source, two
orders, clean plus five 5% corruptions, 25% expert arrivals. Only existing
expert features are used; no new nonexpert capture or confirmation evaluation.
All sources have historical exposure; A's persistent trajectory may have
encountered outer sources without quality labels. Holdout applies to readout
supervision and normalization, not first-ever pipeline exposure.

Keep same-domain TA checkpoints Vid fbb1ed88 / HC ee72f0d9, original Paper48
two-offset pixels, final sixth hidden, Inside[512:768] -> P (256D),
Endpoint[0:512] -> R (512D), A's Uniform Rank-RKL 1792-parameter trajectory
(Vid K1 / HC K8), Old8, Expanded32, A8 and frozen L32 winner W. M-all trains
on all32 candidate rows; M-role on ordered [A,W], retaining no-op duplicates.
P and R heads are shared across A/W; no role feature or separate role heads.
Weighted MSE + alpha*||coefficient||^2, source weights total1, bias unpenalized.
Each inner/outer fit recomputes training-only weighted mean/std; std<1e-8 ->1.

## Prelocked nested selection

Outer folds: same maximum-source LOSO folds as the predecessor, 16 Vid /14 HC.
ALL conditions/orders/candidates from outer source i are excluded from fits,
normalization and alpha selection. Each remaining source is held out once in
an inner LOSO; inner fits exclude both outer and inner source. No row CV.

Alpha grid, ascending: [0.001,0.01,0.1,1,10,100,1000]. Select P/R separately
per dataset, population and outer fold. For BOTH training populations, the
inner objective is equal-source mean of A/W absolute errors (equal roles,
duplicates retained), condition -> order -> source aggregation, over ALL six
conditions. This is the stated deployment A/W generalization objective, not
M-all's pooled32 error. Exact ties choose the smallest alpha. No clipping
during fitting/selection/regression. No target confirmation or outer labels
choose alpha. Then refit each head on all outer-training sources.

There are 422 ordered outer/inner source pairs, two populations, two heads,
seven alphas: 11816 inner candidate ridge fits plus120 selected outer fits.
An eigendecomposition may serve the seven alphas for the same training data;
this changes computation reuse, not ridge equations or representation. No
symmetry reuse across outer folds; all 11816 candidates are evaluated.
Exactly 1808 eigendecompositions (1688 inner population/head plus120 outer).
Fixed controls reuse all/nmax and role/nmax sealed predecessor predictions.

Per-outer file-access guard allows labelled packs only from outer training
sources; no preloaded global labelled pack. Training arrays exclude the inner
source. Inner validation labels are authorized for MAE selection. After all
120 outer fits and selections are sealed, a separate GT-free process reads
unlabelled outer features, predicts A/W and seals outputs. Only thereafter
does evaluation join predecessor search truth/metrics. Integrity hash reads
before fitting are disclosed; they do not expose labels to fitted arrays.

## Endpoints and interpretation

Primary: same-cell source-held-out corruption raw P_A/R_A/P_W/R_W MAE, R2,
Pearson, nested minus fixed-alpha maximum-source LOSO. All/clean/conditions/
orders, raw out-of-range fractions, inner curves and selected-alpha frequency
are also reported. M-all and M-role are both retained; no population winner
selection from outer outcomes. Inner all-condition MAE and primary corrupt
MAE deliberately differ; disclose this fixed choice.

Secondary: clip P/R only in F=PR/(P+R-PR), zero denominator ->0; accept the
same eligible W iff deltaF>0. Preserve no-op/neutral, helpful/harmful/severe
accepts, AUROC/BA and cached expert delta vIoU. No gate/threshold changes or
new winner. Expert-subset gains are not full-stream or future-adaptation gains.

Paired whole-source10000 bootstrap seed20261003, undefined draws retained;
conditional on fixed OOF predictions, overlapping training folds, no refits
or multiplicity adjustment. R2 is relative to evaluation-label variance,
not necessarily a deployed train-mean baseline. Report positive and negative
cases. Alpha improvement implicates this regularization-selection procedure;
it does not uniquely establish high-dimensional variance as cause. Negative
results cannot prove latent no information, exclude every regularizer, or
authorize low-rank/layerwise/nonlinear experiments. No automatic next study.

## Verification and closure

Meaningful CPU tests for nested exclusions, source weighting, normalization,
tie selection, independent SPD ridge parity, zero/no-op and outer-label
invariance. Independent root auditor recomputes every inner curve/selection
with SciPy SPD solves and every selected outer model, checks physical labels,
fixed-control parity, state/candidate hashes, raw metrics and paired intervals.
Public audit verifies anonymous nested folds, per-source inner curves, alpha
choices and scalar metrics without private latent/model/GT spans.
Publish code/protocol/all anonymous outcomes/plots/negative cases to
Zonglin-He/A, verify remote contents, maintain archive check/snapshot/check.
Zero GPU/backbone/expert/candidate/replay/backward/online/deployment calls.
All predecessors and CURRENT_METHOD remain unchanged.
