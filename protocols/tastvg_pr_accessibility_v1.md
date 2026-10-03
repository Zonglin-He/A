# Target-domain P/R readout accessibility audit v1

Finite CPU diagnostic explicitly authorized after f5e3f2d. Two matched tests:
existing source-fit versus target-search-fit linear readout, and fixed analytic
tIoU with predicted versus oracle P/R components. This is target-supervised
root-cause research, never unsupervised TTA, an online gate or a promotion.

## Frozen setting

Keep TA-STVG official same-domain checkpoints, cached final sixth temporal
decoder hidden, Inside[512:768] -> precision and Endpoint[0:512] -> recall,
all existing 32 candidates including Old8, actual A8 and the previously selected
L32 winner. A's 1792 spatial parameter path, Vid K1 / HC K8, original Paper48
two-offset sampling/pixels, 25% expert schedule and corruption stay unchanged.
No backbone, expert, new candidate, replay, backward, GPU or temporal/spatial
model update. Production CURRENT_METHOD and all paused queues remain unchanged.

Original lists are 32 search +16 confirmation sources/dataset, one query/source,
two orders, clean +five 5% corruptions. Existing expert features cover 96 search
and 48 confirmation cells/dataset: Vid 16+8 expert sources, HC 14+7. Only these
288 expert cells have features; do not add the other 864 arrivals to regression
or call the 30 search expert sources 64 independent training sources. Source
probes used 95 Vid /48 HC source train and alpha-selected 31 /16 source validation.
All target panels have historical exposure. New target-search labels now enter
explicitly supervised fitting; confirmation is source-disjoint within this
diagnostic, not fresh or blind in the project history.

## Only new fits: four linear ridge readouts

Use all 32 candidates in each existing search expert cell (3072 rows/dataset,
including retained duplicates, ties and center), with precision and recall
computed from cached physical frame boundaries [fid_i,fid_j+1) and cached search
GT span. Same source-total-weight-one normalization and ridge equation as atlas.
Fit the weighted training mean/std (std<1e-8 ->1), unpenalized intercept and ridge
weights on search only; no confirmation features/labels participate in fitting.
Same normalization *procedure* does not mean freezing source normalization
*statistics*: target-search statistics are re-estimated by that identical
procedure, which is part of the training-domain intervention. This is not a
claim to isolate coefficient changes while freezing source means/variances.

In normalized coordinates, objective is weighted mean square error + alpha
times squared coefficient norm. Preserve original source-selected alphas:
Vid precision=1, recall=1; HC precision=1, recall=.1. No alpha selection, source
refit, additional probe, label shuffle fit, MLP, listwise objective or gate fit.
Keep raw ridge outputs for main regression metrics. Existing source predictions
must reproduce frozen atlas arrays; new fit weights/normalizers stay private.

Separate processes: prepare/pin; search-only fit; GT-free readout; then diagnose.
Fit guard prohibits confirmation payloads and labels, old target labelled ROWS,
raw annotation/media access. After both datasets' four fits are frozen, read
all search/confirm *unlabelled* cached features and seal all scalar predictions
and fixed A/W indices. Only then join confirmation labels. Prior labels and
confirmation results are historically known, so seals establish this execution
order, not first-ever GT blindness. Hash-only integrity reads are disclosed.

## Regression first, analytic decision second

Report source-balanced raw R2/MAE/MSE/Pearson rho for P_A,R_A,P_W,R_W, plus all-32
candidate pooled metrics and training in-sample metrics. Main held-out result
is confirmation corruption; clean and both orders reported separately. Record
clipping fractions. Predictions are clipped only for the fixed analytic map:
F(P,R)=PR/(P+R-PR), with F(0,0)=0 and P,R clipped to [0,1].
Compute delta_F=F(P_W,R_W)-F(P_A,R_A). A genuine fixed L32 replacement is
diagnostically accepted iff delta_F>0 (strict zero, no learned threshold);
no-op stays A. Primary helpful/harmful labels are cached true delta_t greater
than +/-1e-12; neutral and no-op retained, excluded from binary discrimination.
Report oriented delta_F AUROC without sign flipping, source-balanced TPR/FPR,
balanced accuracy, precision/accept counts, within-source AUC, severe accepted
v-harm <-.05, and cached expert-subset t/v deltas of this offline decision.
These are diagnostic simulations on frozen candidates, not new online outputs
or full-flow gains. The fixed winner is never reselected by the new P/R model.

Prelock six ladder arms for *both* source-fit and target-search-fit predictions:
all predicted; GT anchor P_A/R_A; GT winner P_W/R_W; GT precision P_A/P_W;
GT recall R_A/R_W; all GT. The latter must reconstruct every cached A/W tIoU
and primary binary decision as a positive control. Partial replacements can
mix mutually incompatible P/R and are component diagnostics, not valid interval
quality models. No choice of ladder or setting on confirmation.

## Statistics and limits

Cell -> condition -> order -> source aggregation; source has equal total weight.
Class-balanced AUROC follows predecessor: each helpful source has weight1
within helpful class, each harmful source weight1 within harmful class. Bootstrap
whole sources 10000 times, seed20261003, paired for source-fit vs target-fit and
ladder comparisons. Class-free or zero-variance draws stay undefined, never .5
or 0. Per-order point estimates and delete-one-source influence are not refits.
All source/clean/negative/neutral results retained, exploratory CIs unadjusted.

Interpretation cannot uniquely identify representation, transfer or objective:
changing training domain also changes sample size, candidate/anchor distribution,
corruption mix and normalizer. Target-fit recovery supports local mapping
adaptability, not unsupervised recoverability or proof that transfer is the
sole cause. Failure at one fixed source-selected alpha, one linear family and
16/14 training sources does not prove no latent information. P/R algebra is
exact for true interval tIoU, not sufficient for vIoU with fixed space errors.
Only compare frozen candidate decisions; no automatic layerwise/expert followup.

## Verification / public closure

CPU tests: exact P/R identity, empty overlap, clipping, monotonicity, raw-vs-
clipped distinction, weighted ridge against independent solve, replication
invariance, source normalization, no-op and undefined classes. Root audit refits
using an independent linear solver; checks all payload/label hashes, source
disjointness, frozen prediction parity, fit/readout/label seal ordering, A/W,
all oracle algebra, weighted metrics/bootstraps and source influence. Public
audit recalculates anonymous scalar rows without private features, weights or
GT spans. Publish code, configuration/protocol, full anonymous results/negative
cases and figures to Zonglin-He/A; exclude weights, latents, raw GT/media.
Verify remote bytes, update RESEARCH_HISTORY, run check/snapshot/check and save
FINAL_COMPLETION. No scientific mechanism is promoted from this diagnostic.
