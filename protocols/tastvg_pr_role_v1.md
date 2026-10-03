# Role-conditioned P/R readout audit v1

Authorized CPU diagnostic after d408f154: change only the target-search
training candidate population, comparing the frozen M-all with M-role trained
on the same cell's A8 anchor and already frozen L32 winner W. No new gate,
threshold, alpha search, representation, model inference or deployment update.

## Frozen inputs and actual cohort

Reuse all 288 expert cells and source lists from tastvg_pr_accessibility_v1.
Original design is 32 search +16 confirm sources/dataset, one query/source,
two orders, clean plus five 5% corruptions, 25% expert arrivals. Existing
expert caches actually cover Vid 16 search /8 confirm sources and HC2 14 /7,
96 search +48 confirm cells/dataset, 240 corrupt +48 clean cells in total.
All sources have historical exposure; confirmation is source-disjoint within
this diagnostic, not fresh. Do not add the 864 nonexpert arrivals to regressions.

Same official same-domain TA-STVG checkpoints (Vid fbb1ed88, HC2 ee72f0d9),
final sixth temporal decoder hidden, Inside[512:768] -> P and
Endpoint[0:512] -> R. A's Uniform Rank-RKL 1792 spatial parameters, Vid K1 /HC
K8, original Paper48 two-offset pixels, expert schedule, A8, all 32 intervals
and old L32 winner remain fixed. Native is not substituted for A8. New quality
models never select a new W or change future states.

## Exactly two training populations

M-all: reuse predecessor's target-search model, fitted on all 32 candidates
per cell, 3072 training rows/dataset. Recompute its unlabelled predictions and
require scalar parity with the sealed predecessor; no scientific M-all refit.

M-role: train one shared P ridge and one shared R ridge per dataset on the
ordered two rows [A, W] per search cell, 192 rows/dataset. No separate A/W
heads, role ID input or relative objective. Include clean/corrupt, ties and
no-op cells without GT-based filtering. If A=W, retain both identical rows:
each cell still contributes one anchor and one winner role with equal weight.

Same equal-source weighted ridge mean-square objective and unpenalized bias,
same weighted training mean/std procedure (std<1e-8 ->1); statistics are
re-estimated on the selected training population. Thus the intervention
includes candidate diversity, label distribution and fitted normalizer/bias,
not just coefficients under frozen M-all statistics. Equal-source weights
sum to 1 in both arms, so reducing row count does not automatically rescale
the fixed ridge penalty. Preserve old source-selected alpha: Vid P=1/R=1,
HC P=1/R=.1. Four new scientific CPU fits; independent verification refits
are accounted separately. No alpha/temperature/threshold selection.

Fit P/R truth from the same cached search GT spans and physical candidate
boundaries [frame_i, frame_j+1). Search fitting is explicitly target-supervised
root-cause research, not a legal unsupervised TTA result. Fitting excludes all
confirmation features/labels and previous labelled ROWS. Prepare only reads
metadata and integrity hashes. Separate processes prepare -> fit -> readout ->
diagnose. Freeze four models, then predict both models on all existing cached
cells without GT, seal all scores and fixed A/W choices, then join confirmation
GT/cached metrics. Historical exposure is disclosed; sealing does not imply
first-ever blindness.

## Measurements and fixed analytic decision

Primary held-out panel: confirmation corruption. First report raw unbounded
P_A, R_A, P_W, R_W MAE, R2 and Pearson correlation; MSE, means, clipping
fractions and true variance are supporting diagnostics. All-32 pooled metrics
and search in-sample fitting are separate, not deployment-role correctness.
Different population variances can change R2 without equivalently changing
absolute error. Do not infer causality solely from pooled-versus-role R2.

For both models clip P/R to [0,1] only in F(P,R)=PR/(P+R-PR), F(0,0)=0.
Accept the same frozen W iff eligible and F(W)-F(A)>0, strict zero. No-op
retains A. Helpful/harmful uses cached true delta_t > +/-1e-12; neutral/no-op
remain in results and are excluded from binary metrics. Report AUROC with
fixed orientation, balanced accuracy/TPR/FPR/precision, accept counts,
accepted severe delta_v<-.05 and cached expert-subset t/v gain/loss. These
are offline readout simulations, not a new full online stream or new TTA gain.
No new oracle ladder variants; inherited true tIoU identity is only an
arithmetic check. Keep clean, both orders, source influence and negative cases.

Aggregate condition -> order -> equal source; for binary AUROC use equal
source weight within each class, as predecessor. Paired whole-source 10000
bootstrap, seed 20261003; preserve undefined draws and single-class order
metrics. Delete-one-source is influence of frozen predictions, not refitting.
Intervals are exploratory and not multiple-comparison-adjusted. MAE/R2/rho
paired role-minus-all and decision/utility differences use identical draws.

Recovery would support a population-conditioned mapping effect in this
family, not uniquely prove selection-induced mismatch or unsupervised
recoverability. Failure is scoped to one linear family, fixed source alpha,
14/16 search sources and the cached representation; do not claim latent lacks
information or automatically run layerwise/MLP/motion/expert experiments.
No method promotion based on this target-supervised audit.

## Closure

Run meaningful CPU controls; independent SciPy solve verifies both reused
M-all and new M-role fits, all immutable payloads, predictions, physical
labels, role membership, accept rules, metrics/bootstrap and state hashes.
Public audit uses anonymous scalar outputs without private features/weights/
GT spans. Publish code/protocol/all results/negative cases/figures to
Zonglin-He/A, verify remote bytes, record archive check/snapshot/check.
Exclude coefficients, normalizers, latents, private media/annotations,
checkpoint weights and conversation records. Preserve all predecessor assets.
