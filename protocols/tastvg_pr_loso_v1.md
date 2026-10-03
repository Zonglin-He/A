# Search-domain source-held-out P/R readout audit

User-authorized CPU diagnosis after c1558266eaefee13b745202f332f7f52bffa1748.
Question: do the same final-layer linear P/R readouts generalize to a source
excluded from all of their supervised training and fitted normalization?
Do not declare high dimensionality, memorization or representation failure a
unique cause from this experiment alone.

## Fixed setting

Reuse predecessor search expert cells only: 96 per dataset, Vid 16 independent
sources and HC-STVG-v2 14. Each panel has 80 corrupt cells (five 5% conditions)
and 16 clean cells. Original design was 32 search sources, one query/source,
two orders, six conditions, 25% expert schedule; only the existing expert
features are available. No additional nonexpert capture. All sources have
historical exposure. A's cached persistent states may have encountered a held
source without quality labels: this is readout-source holdout, not first-ever
pipeline exposure or fresh test. Confirmation is neither refitted nor newly
evaluated in this run; prior confirmation scores are contextual only.

Keep same-domain TA checkpoints Vid fbb1ed88 /HC ee72f0d9, original Paper48
two-offset pixels, sixth temporal hidden Inside[512:768] -> P and
Endpoint[0:512] -> R, A's 1792 Uniform Rank-RKL state (Vid K1 /HC K8), Old8,
Expanded32, A8 and the frozen L32 winner W. Zero new GPU, backbone, expert,
candidate, replay, backward, online stream or deployment calls.

Two training populations: M-all uses all32 candidate rows per training cell;
M-role uses the ordered [A,W] rows, retaining duplicate no-op roles. Shared
P/R heads, no role feature or separate A/W heads. Same equal-source weighted
MSE + alpha*||coefficient||^2, weights sum1, unpenalized bias. Alpha remains
Vid P1/R1 and HC P1/R.1, selected in the original source study. Within every
fold/population, recompute mean/std/bias on training sources only; std<1e-8
becomes1. No dimensionality, alpha, layer, loss or threshold search.

## Prelocked folds and learning curve

For each of the 30 search sources, hold out ALL of its conditions, orders and
candidate roles. Rank remaining source IDs by SHA256 of
`tastvg_pr_loso_v1|dataset|held_id|train_id`. Take nested prefixes of 4,8,12
and all remaining sources. Thus the maximum is 15 for Vid /13 for HC, never
the full training population containing the tested source. Exactly one
deterministic nested list per held-source fold; no subset winner selection,
random repeats or confirmation selection. The same lists serve M-all/M-role
and P/R. Models at each source count predict the same held source, allowing
paired curves. Counts vary in candidate rows because sources can have
different numbers of cached expert cells; source weight remains equal.

There are 30 folds *4 source counts *2 populations *2 heads =480 new CPU
scientific fits. Reuse four prior heads per population across the two
datasets: 8 prior heads, without scientific
refitting, for in-sample predictions. Independent verification refits are
reported separately. No new full-source fit and no model promotion.

Prepare locks metadata, folds, prior models and private feature payload hashes.
Extraction reads authorized search GT and writes a separate private training
pack per source. During each fit an audit hook denies opening that fold's
held-source labelled pack, all confirmation features/labels, prior ROWS and
raw media/annotations. All 480 models seal, then a separate GT-free readout
predicts held-source features and seals all scalar outputs before cached search
GT/metrics join for evaluation. Integrity hash reads precede access guards and
are disclosed. Target search supervision is diagnostic, not unsupervised TTA.

## Endpoints and uncertainty

Primary: source-held-out search corruption raw P_A/R_A/P_W/R_W MAE, MSE, R2,
Pearson correlation for both populations; compare maximum-count OOF versus
reused in-sample readouts on exactly the same cells. Curve counts4/8/12/max
are descriptive, not parameter selection. Preserve raw unbounded regression;
truth variance/means and clipping fractions are supporting diagnostics.
Source-balanced R2 uses the SAME evaluation labels across regimes; negative
R2 means worse than a test-label grand-mean oracle, not necessarily worse than
a deployed train-mean predictor. Do not compare R2 across different cohorts
as a causal test. A/R role diversity and fitted normalizers differ between
populations, so an M-role contrast is not an isolated row-count intervention.

Secondary fixed readout: clip P/R only for F=PR/(P+R-PR), F(0,0)=0; accept
the already frozen eligible W iff predicted F(W)-F(A)>0. Report AUROC,
balanced accuracy, helpful/harmful/severe accepts and cached expert-subset
delta vIoU. No threshold scan, new winner or oracle ladder. Counts are cells,
aggregation condition -> order -> equal source; binary metrics equal source
weight within class. Clean, both orders, condition panels and success/failure
cases are retained. No full-stream gain claim or multiplication by25%.

Paired whole-source10000 bootstrap, seed20261003, undefined draws retained.
The bootstrap resamples already fixed OOF prediction/label clusters, without
refitting folds; intervals are conditional on these overlapping training folds
and one locked subset sequence. They are not full repeated-CV uncertainty.
Exploratory pointwise intervals are not multiplicity-adjusted. A short curve
may be nonmonotone; it does not establish a plateau or a larger-data ceiling.
Poor LOSO supports source-level generalization failure within this fixed
family, not proof of literal memorization, dimensionality as unique cause,
absence of latent information, or inevitability of a nonlinear/layerwise fix.

## Closure

Meaningful CPU leakage/normalizer/nesting/weighting/identity tests; independent
SciPy solve checks all480 fits and scalar outputs. Independent public audit
checks all anonymous raw predictions, labels, fixed decisions, aggregation,
paired intervals and source lists without private features/weights/GT spans.
Publish code, protocol, anonymous results, negatives and plots to Zonglin-He/A;
verify remote bytes and maintain RESEARCH_HISTORY check/snapshot/check.
Private packs, models, normalizers, latents, spans, media and original weights
stay excluded. A/CURRENT and every predecessor asset remain unchanged.
